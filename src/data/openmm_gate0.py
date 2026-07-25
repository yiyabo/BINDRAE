"""OpenMM Gate-0 path contracts and coordinate-mapping utilities.

The functions in this module keep model-path identity explicit before any
force-field calculation.  OpenMM itself is imported only by the executable
scorer so these validation helpers remain usable in the regular BINDRAE
environment and in unit tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.data.residue_identity import (
    ResidueKey,
    normalize_residue_key,
    residue_keys_from_array,
)
from src.stage1.data.residue_constants import (
    restype_1to3,
    restype_name_to_atom14_names,
    restypes,
)


PATH_CANDIDATE_SCHEMA_VERSION = "bindrae_stage2_path_candidate_v3"
PATH_CANDIDATE_LEGACY_SCHEMA_VERSIONS = (
    "bindrae_stage2_path_candidate_v2",
)
FRAME_REFERENCE_CACHE_SCHEMA_VERSION = "bindrae_path4_frame_reference_cache_v1"
OPENMM_GATE0_SCORE_SCHEMA_VERSION = "bindrae_path4_openmm_gate0_score_v5"
OPENMM_GATE0_RELAXED_VALIDITY_SCHEMA_VERSION = (
    "bindrae_path4_relaxed_frame_validity_v1"
)
PATH4_GATE0_PAIRED_SUMMARY_SCHEMA_VERSION = (
    "bindrae_path4_gate0_paired_summary_v2"
)
RELAXED_VALIDITY_METRIC = "protein_residue_net_force_max_kj_mol_nm"


class ReferenceTopologyStateError(ValueError):
    """Physical preflight failure with the measured state attached."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def reconstruct_peptide_carbonyl_oxygen(
    atom14_pos_angstrom: np.ndarray,
    atom14_mask: np.ndarray,
    node_mask: np.ndarray,
    peptide_bond_mask: np.ndarray,
    *,
    bond_length_angstrom: float = 1.231,
) -> Tuple[np.ndarray, int]:
    """Place peptide carbonyl O from the current CA-C-N(next) plane.

    The Stage-2 state fixes each residue frame and chi angles but does not carry
    an explicit carbonyl-O degree of freedom.  The OpenFold FK template used by
    the model has a frame-convention mismatch for O, so physical exports rebuild
    that atom from peptide geometry instead of injecting the mismatched template
    coordinate into OpenMM.
    """
    positions = np.asarray(atom14_pos_angstrom, dtype=np.float64)
    mask = np.asarray(atom14_mask, dtype=np.bool_)
    nodes = np.asarray(node_mask, dtype=np.bool_)
    peptide = np.asarray(peptide_bond_mask, dtype=np.bool_)
    if positions.ndim != 4 or positions.shape[-2:] != (14, 3):
        raise ValueError("atom14_pos_angstrom must have shape [T, N, 14, 3]")
    if mask.shape != positions.shape[:-1]:
        raise ValueError("atom14_mask does not match atom14 positions")
    if nodes.shape != (positions.shape[1],):
        raise ValueError("node_mask does not match the residue axis")
    if peptide.shape != (max(positions.shape[1] - 1, 0),):
        raise ValueError("peptide_bond_mask does not match the residue axis")
    if bond_length_angstrom <= 0.0:
        raise ValueError("bond_length_angstrom must be positive")
    if positions.shape[1] < 2:
        return positions.astype(atom14_pos_angstrom.dtype, copy=True), 0

    valid = (
        nodes[:-1][None, :]
        & nodes[1:][None, :]
        & peptide[None, :]
        & mask[:, :-1, 1]
        & mask[:, :-1, 2]
        & mask[:, :-1, 3]
        & mask[:, 1:, 0]
    )
    frame_indices, residue_indices = np.nonzero(valid)
    if frame_indices.size == 0:
        return positions.astype(atom14_pos_angstrom.dtype, copy=True), 0

    carbon = positions[frame_indices, residue_indices, 2]
    toward_ca = positions[frame_indices, residue_indices, 1] - carbon
    toward_next_n = positions[frame_indices, residue_indices + 1, 0] - carbon
    ca_norm = np.linalg.norm(toward_ca, axis=-1, keepdims=True)
    n_norm = np.linalg.norm(toward_next_n, axis=-1, keepdims=True)
    if np.any(ca_norm <= 1e-6) or np.any(n_norm <= 1e-6):
        raise ValueError("Degenerate CA-C-N geometry while rebuilding carbonyl O")
    bisector = -(toward_ca / ca_norm + toward_next_n / n_norm)
    bisector_norm = np.linalg.norm(bisector, axis=-1, keepdims=True)
    if np.any(bisector_norm <= 1e-6):
        raise ValueError("Degenerate peptide bisector while rebuilding carbonyl O")

    rebuilt = positions.copy()
    rebuilt[frame_indices, residue_indices, 3] = (
        carbon + float(bond_length_angstrom) * bisector / bisector_norm
    )
    return rebuilt.astype(atom14_pos_angstrom.dtype, copy=False), int(frame_indices.size)


@dataclass(frozen=True)
class PathCandidate:
    schema_version: str
    sample_id: str
    candidate_label: str
    path_parameterization: str
    times: np.ndarray
    atom14_pos_angstrom: np.ndarray
    atom14_mask: np.ndarray
    aatype: np.ndarray
    node_mask: np.ndarray
    residue_keys: Tuple[ResidueKey, ...]
    residue_identity_hash: str
    rigid_rotation_matrix: np.ndarray | None
    rigid_translation_angstrom: np.ndarray | None
    chi_radians: np.ndarray | None
    source_path: Path

    @property
    def n_frames(self) -> int:
        return int(self.times.size)

    @property
    def n_residues(self) -> int:
        return int(self.aatype.size)

    @property
    def has_product_state(self) -> bool:
        return all(
            value is not None
            for value in (
                self.rigid_rotation_matrix,
                self.rigid_translation_angstrom,
                self.chi_radians,
            )
        )


@dataclass(frozen=True)
class TopologyAtomRecord:
    index: int
    residue_index: int
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atom_name: str
    element: str

    @property
    def residue_key(self) -> ResidueKey:
        return normalize_residue_key(
            (self.chain_id, self.residue_number, self.insertion_code)
        )


@dataclass(frozen=True)
class CandidateTopologyMapping:
    candidate_residue_indices: np.ndarray
    candidate_atom14_indices: np.ndarray
    topology_atom_indices: np.ndarray
    topology_residue_indices: np.ndarray
    residue_all_atom_indices: Tuple[np.ndarray, ...]
    residue_mapped_topology_indices: Tuple[np.ndarray, ...]
    residue_mapped_atom14_indices: Tuple[np.ndarray, ...]
    mapped_ca_topology_indices: np.ndarray
    mapped_ca_candidate_residue_indices: np.ndarray
    mapped_residue_fraction: float
    mapped_atom_fraction: float
    ignored_chain_labels: bool


@dataclass(frozen=True)
class FrameReferenceCache:
    sample_id: str
    residue_identity_hash: str
    times: np.ndarray
    all_atom_pos_angstrom: np.ndarray
    source_candidate: str
    source_candidate_sha256: str
    system_contract: Dict[str, Any]
    generation_contract: Dict[str, Any]
    frame_preflight: Tuple[Dict[str, Any], ...]
    source_path: Path


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"Expected scalar text, got shape={array.shape}")
    return str(array.item())


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def relaxed_frame_validity_contract(
    maximum_residue_net_force_kj_mol_nm: float,
) -> Dict[str, Any]:
    threshold = float(maximum_residue_net_force_kj_mol_nm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Relaxed-frame residue-force threshold must be positive")
    return {
        "schema_version": OPENMM_GATE0_RELAXED_VALIDITY_SCHEMA_VERSION,
        "metric": RELAXED_VALIDITY_METRIC,
        "maximum_allowed_kj_mol_nm": threshold,
        "scope": "protein_residue_net_force_after_restrained_minimization",
        "severe_clash_is_reported_separately": True,
    }


def assess_relaxed_frame_validity(
    protein_residue_net_force_max_kj_mol_nm: float,
    *,
    maximum_residue_net_force_kj_mol_nm: float,
) -> Dict[str, Any]:
    contract = relaxed_frame_validity_contract(
        maximum_residue_net_force_kj_mol_nm
    )
    observed = float(protein_residue_net_force_max_kj_mol_nm)
    reasons: List[str] = []
    if not math.isfinite(observed):
        reasons.append("non_finite_protein_residue_net_force")
    elif observed > float(contract["maximum_allowed_kj_mol_nm"]):
        reasons.append("protein_residue_net_force_above_threshold")
    return {
        **contract,
        "observed_kj_mol_nm": observed,
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


def build_frame_reference_cache_payload(
    candidate: PathCandidate,
    all_atom_pos_angstrom: np.ndarray,
    *,
    source_candidate_sha256: str,
    system_contract: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
    frame_preflight: Sequence[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    positions = np.asarray(all_atom_pos_angstrom, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[0] != candidate.n_frames:
        raise ValueError("Frame reference positions must have shape [T, A, 3]")
    if positions.shape[1] == 0 or positions.shape[2] != 3:
        raise ValueError("Frame reference cache has an invalid atom axis")
    if not np.isfinite(positions).all():
        raise ValueError("Frame reference cache contains non-finite positions")
    if len(frame_preflight) != candidate.n_frames:
        raise ValueError("Frame reference preflight does not match the time grid")
    if len(source_candidate_sha256) != 64:
        raise ValueError("source_candidate_sha256 must be a SHA-256 hex digest")
    try:
        int(source_candidate_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_candidate_sha256 must be hexadecimal") from exc

    return {
        "schema_version": np.array(FRAME_REFERENCE_CACHE_SCHEMA_VERSION),
        "sample_id": np.array(candidate.sample_id),
        "residue_identity_hash": np.array(candidate.residue_identity_hash),
        "times": np.asarray(candidate.times, dtype=np.float64),
        "all_atom_pos_angstrom": positions,
        "topology_atom_count": np.array(positions.shape[1], dtype=np.int64),
        "source_candidate": np.array(str(candidate.source_path)),
        "source_candidate_sha256": np.array(source_candidate_sha256),
        "system_contract_json": np.array(_canonical_json(system_contract)),
        "generation_contract_json": np.array(_canonical_json(generation_contract)),
        "frame_preflight_json": np.array(
            json.dumps(list(frame_preflight), sort_keys=True, separators=(",", ":"))
        ),
    }


def load_frame_reference_cache(
    path: Path,
    candidate: PathCandidate,
    *,
    system_contract: Mapping[str, Any],
    topology_atom_count: int,
) -> FrameReferenceCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "schema_version",
            "sample_id",
            "residue_identity_hash",
            "times",
            "all_atom_pos_angstrom",
            "topology_atom_count",
            "source_candidate",
            "source_candidate_sha256",
            "system_contract_json",
            "generation_contract_json",
            "frame_preflight_json",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Frame reference cache {path} misses fields: {missing}")
        schema = _scalar_text(loaded["schema_version"])
        if schema != FRAME_REFERENCE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported frame reference cache schema {schema!r}; "
                f"expected {FRAME_REFERENCE_CACHE_SCHEMA_VERSION!r}"
            )
        sample_id = _scalar_text(loaded["sample_id"])
        if sample_id != candidate.sample_id:
            raise ValueError(
                f"Frame reference sample mismatch: {sample_id} != {candidate.sample_id}"
            )
        residue_hash = _scalar_text(loaded["residue_identity_hash"])
        if residue_hash != candidate.residue_identity_hash:
            raise ValueError("Frame reference residue identity hash mismatch")
        times = np.asarray(loaded["times"], dtype=np.float64)
        if times.shape != candidate.times.shape or not np.allclose(
            times, candidate.times, atol=1e-8, rtol=0.0
        ):
            raise ValueError("Frame reference time grid mismatch")
        observed_atom_count = int(np.asarray(loaded["topology_atom_count"]).item())
        if observed_atom_count != int(topology_atom_count):
            raise ValueError(
                "Frame reference topology atom count mismatch: "
                f"{observed_atom_count} != {int(topology_atom_count)}"
            )
        positions = np.asarray(loaded["all_atom_pos_angstrom"], dtype=np.float64)
        expected_shape = (candidate.n_frames, int(topology_atom_count), 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"Frame reference position shape mismatch: {positions.shape} != {expected_shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("Frame reference cache contains non-finite positions")
        observed_system_contract = json.loads(
            _scalar_text(loaded["system_contract_json"])
        )
        if _canonical_json(observed_system_contract) != _canonical_json(system_contract):
            raise ValueError("Frame reference implicit-system contract mismatch")
        generation_contract = json.loads(
            _scalar_text(loaded["generation_contract_json"])
        )
        frame_preflight_value = json.loads(
            _scalar_text(loaded["frame_preflight_json"])
        )
        if not isinstance(frame_preflight_value, list) or len(frame_preflight_value) != candidate.n_frames:
            raise ValueError("Frame reference preflight record count mismatch")
        frame_preflight = tuple(dict(value) for value in frame_preflight_value)
        return FrameReferenceCache(
            sample_id=sample_id,
            residue_identity_hash=residue_hash,
            times=times,
            all_atom_pos_angstrom=positions,
            source_candidate=_scalar_text(loaded["source_candidate"]),
            source_candidate_sha256=_scalar_text(
                loaded["source_candidate_sha256"]
            ),
            system_contract=dict(observed_system_contract),
            generation_contract=dict(generation_contract),
            frame_preflight=frame_preflight,
            source_path=path,
        )


def load_path_candidate(path: Path) -> PathCandidate:
    path = Path(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "schema_version",
            "sample_id",
            "candidate_label",
            "path_parameterization",
            "times",
            "atom14_pos_angstrom",
            "atom14_mask",
            "aatype",
            "node_mask",
            "residue_keys",
            "residue_identity_hash",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Path candidate {path} misses fields: {missing}")
        schema = _scalar_text(loaded["schema_version"])
        supported_schemas = {
            PATH_CANDIDATE_SCHEMA_VERSION,
            *PATH_CANDIDATE_LEGACY_SCHEMA_VERSIONS,
        }
        if schema not in supported_schemas:
            raise ValueError(
                f"Unsupported path candidate schema {schema!r}; "
                f"expected one of {sorted(supported_schemas)!r}"
            )
        state_fields = {
            "rigid_rotation_matrix",
            "rigid_translation_angstrom",
            "chi_radians",
        }
        if schema == PATH_CANDIDATE_SCHEMA_VERSION:
            missing_state = sorted(state_fields - set(loaded.files))
            if missing_state:
                raise ValueError(
                    f"Path candidate {path} misses v3 product-state fields: "
                    f"{missing_state}"
                )
        times = np.asarray(loaded["times"], dtype=np.float64)
        positions = np.asarray(loaded["atom14_pos_angstrom"], dtype=np.float64)
        mask = np.asarray(loaded["atom14_mask"], dtype=np.bool_)
        aatype = np.asarray(loaded["aatype"], dtype=np.int64)
        node_mask = np.asarray(loaded["node_mask"], dtype=np.bool_)
        residue_keys = tuple(residue_keys_from_array(loaded["residue_keys"]))
        candidate = PathCandidate(
            schema_version=schema,
            sample_id=_scalar_text(loaded["sample_id"]),
            candidate_label=_scalar_text(loaded["candidate_label"]),
            path_parameterization=_scalar_text(loaded["path_parameterization"]),
            times=times,
            atom14_pos_angstrom=positions,
            atom14_mask=mask,
            aatype=aatype,
            node_mask=node_mask,
            residue_keys=residue_keys,
            residue_identity_hash=_scalar_text(loaded["residue_identity_hash"]),
            rigid_rotation_matrix=(
                np.asarray(loaded["rigid_rotation_matrix"], dtype=np.float64)
                if "rigid_rotation_matrix" in loaded
                else None
            ),
            rigid_translation_angstrom=(
                np.asarray(loaded["rigid_translation_angstrom"], dtype=np.float64)
                if "rigid_translation_angstrom" in loaded
                else None
            ),
            chi_radians=(
                np.asarray(loaded["chi_radians"], dtype=np.float64)
                if "chi_radians" in loaded
                else None
            ),
            source_path=path,
        )
    validate_path_candidate(candidate)
    return candidate


def validate_path_candidate(candidate: PathCandidate) -> None:
    if candidate.times.ndim != 1 or candidate.times.size < 3:
        raise ValueError("Path candidate requires a one-dimensional >=3-frame time grid")
    if not np.isclose(candidate.times[0], 0.0, atol=1e-7):
        raise ValueError("Path candidate does not start at t=0")
    if not np.isclose(candidate.times[-1], 1.0, atol=1e-7):
        raise ValueError("Path candidate does not end at t=1")
    if np.any(np.diff(candidate.times) <= 0.0):
        raise ValueError("Path candidate times are not strictly increasing")
    expected_positions = (candidate.n_frames, candidate.n_residues, 14, 3)
    expected_mask = expected_positions[:-1]
    if candidate.atom14_pos_angstrom.shape != expected_positions:
        raise ValueError(
            "atom14_pos_angstrom shape mismatch: "
            f"{candidate.atom14_pos_angstrom.shape} != {expected_positions}"
        )
    if candidate.atom14_mask.shape != expected_mask:
        raise ValueError(
            f"atom14_mask shape mismatch: {candidate.atom14_mask.shape} != {expected_mask}"
        )
    if candidate.node_mask.shape != (candidate.n_residues,):
        raise ValueError("node_mask does not match the residue axis")
    if len(candidate.residue_keys) != candidate.n_residues:
        raise ValueError("residue_keys do not match the residue axis")
    if np.any((candidate.aatype < 0) | (candidate.aatype >= len(restypes))):
        raise ValueError("Path candidate contains unsupported amino-acid indices")
    active = candidate.atom14_mask & candidate.node_mask[None, :, None]
    if not active.any():
        raise ValueError("Path candidate has no active atoms")
    if not np.isfinite(candidate.atom14_pos_angstrom[active]).all():
        raise ValueError("Path candidate contains non-finite active coordinates")
    endpoint_backbone = active[[0, -1], :, :3]
    if not endpoint_backbone.all(axis=-1).any():
        raise ValueError("Path candidate has no complete endpoint backbone mapping")

    if candidate.schema_version == PATH_CANDIDATE_SCHEMA_VERSION:
        if not candidate.has_product_state:
            raise ValueError("Path candidate v3 requires complete frame/chi state")
        rotation = np.asarray(candidate.rigid_rotation_matrix)
        translation = np.asarray(candidate.rigid_translation_angstrom)
        chi = np.asarray(candidate.chi_radians)
        expected_rotation = (candidate.n_frames, candidate.n_residues, 3, 3)
        expected_translation = (candidate.n_frames, candidate.n_residues, 3)
        expected_chi = (candidate.n_frames, candidate.n_residues, 4)
        if rotation.shape != expected_rotation:
            raise ValueError(
                f"rigid_rotation_matrix shape mismatch: {rotation.shape} != "
                f"{expected_rotation}"
            )
        if translation.shape != expected_translation:
            raise ValueError(
                f"rigid_translation_angstrom shape mismatch: {translation.shape} != "
                f"{expected_translation}"
            )
        if chi.shape != expected_chi:
            raise ValueError(
                f"chi_radians shape mismatch: {chi.shape} != {expected_chi}"
            )
        active_residue = candidate.node_mask[None, :]
        if not np.isfinite(rotation[active_residue.repeat(candidate.n_frames, axis=0)]).all():
            raise ValueError("Path candidate contains non-finite active rotations")
        if not np.isfinite(translation[active_residue.repeat(candidate.n_frames, axis=0)]).all():
            raise ValueError("Path candidate contains non-finite active translations")
        if not np.isfinite(chi[active_residue.repeat(candidate.n_frames, axis=0)]).all():
            raise ValueError("Path candidate contains non-finite active chi angles")

        active_rotation = rotation[:, candidate.node_mask]
        identity = np.eye(3, dtype=np.float64)
        orthogonality_error = np.max(
            np.abs(active_rotation @ np.swapaxes(active_rotation, -1, -2) - identity)
        )
        determinant_error = np.max(np.abs(np.linalg.det(active_rotation) - 1.0))
        if orthogonality_error > 2.0e-4 or determinant_error > 2.0e-4:
            raise ValueError(
                "Path candidate rotations are not proper SO(3) matrices: "
                f"orthogonality_error={orthogonality_error:.6g}, "
                f"determinant_error={determinant_error:.6g}"
            )
        ca_valid = candidate.atom14_mask[:, :, 1] & candidate.node_mask[None, :]
        if ca_valid.any():
            frame_ca_error = np.linalg.norm(
                translation[ca_valid] - candidate.atom14_pos_angstrom[:, :, 1][ca_valid],
                axis=-1,
            )
            if float(frame_ca_error.max()) > 1.0e-3:
                raise ValueError(
                    "Path candidate frame translations do not match atom14 CA: "
                    f"max_error={float(frame_ca_error.max()):.6g} A"
                )


def topology_atom_records(topology) -> List[TopologyAtomRecord]:
    records: List[TopologyAtomRecord] = []
    for residue_index, residue in enumerate(topology.residues()):
        try:
            residue_number = int(str(residue.id).strip())
        except ValueError as exc:
            raise ValueError(
                f"OpenMM residue {residue.index} has non-integer id={residue.id!r}"
            ) from exc
        insertion_code = str(getattr(residue, "insertionCode", "") or "").strip()
        chain_id = str(getattr(residue.chain, "id", "") or "").strip()
        for atom in residue.atoms():
            element = ""
            if atom.element is not None:
                element = str(atom.element.symbol).strip().upper()
            records.append(
                TopologyAtomRecord(
                    index=int(atom.index),
                    residue_index=int(residue_index),
                    chain_id=chain_id,
                    residue_number=residue_number,
                    insertion_code=insertion_code,
                    residue_name=str(residue.name).strip().upper(),
                    atom_name=str(atom.name).strip().upper(),
                    element=element,
                )
            )
    return records


def _residue_token(key: ResidueKey, ignore_chain: bool):
    return ("" if ignore_chain else key[0], key[1], key[2])


def _expected_atom_names(aatype: int) -> List[str]:
    residue_letter = restypes[int(aatype)]
    residue_name = restype_1to3[residue_letter]
    return [str(name).upper() for name in restype_name_to_atom14_names[residue_name]]


def build_candidate_topology_mapping(
    candidate: PathCandidate,
    atom_records: Sequence[TopologyAtomRecord],
    *,
    minimum_residue_fraction: float = 0.98,
    minimum_atom_fraction: float = 0.95,
) -> CandidateTopologyMapping:
    if not 0.0 < minimum_residue_fraction <= 1.0:
        raise ValueError("minimum_residue_fraction must be in (0, 1]")
    if not 0.0 < minimum_atom_fraction <= 1.0:
        raise ValueError("minimum_atom_fraction must be in (0, 1]")
    if not atom_records:
        raise ValueError("OpenMM topology has no atoms")

    topology_residue_names: Dict[int, str] = {}
    topology_residue_keys: Dict[int, ResidueKey] = {}
    topology_atoms_by_residue: Dict[int, List[TopologyAtomRecord]] = {}
    for atom in atom_records:
        topology_residue_names[atom.residue_index] = atom.residue_name
        topology_residue_keys[atom.residue_index] = atom.residue_key
        topology_atoms_by_residue.setdefault(atom.residue_index, []).append(atom)

    standard_names = set(restype_name_to_atom14_names)
    standard_aliases = {
        "ASH": "ASP",
        "CYM": "CYS",
        "CYX": "CYS",
        "GLH": "GLU",
        "HID": "HIS",
        "HIE": "HIS",
        "HIP": "HIS",
        "LYN": "LYS",
    }
    protein_residue_indices = [
        residue_index
        for residue_index, name in topology_residue_names.items()
        if name in standard_names or name in standard_aliases
    ]
    if not protein_residue_indices:
        raise ValueError("OpenMM topology has no standard protein residues")
    candidate_chains = {key[0] for key in candidate.residue_keys}
    topology_chains = {
        topology_residue_keys[index][0] for index in protein_residue_indices
    }
    ignore_chain = len(candidate_chains) == 1 and len(topology_chains) == 1

    topology_index_by_token: Dict[Tuple[str, int, str], int] = {}
    for residue_index in protein_residue_indices:
        token = _residue_token(topology_residue_keys[residue_index], ignore_chain)
        if token in topology_index_by_token:
            raise ValueError(f"Duplicate OpenMM protein residue identity: {token}")
        topology_index_by_token[token] = residue_index

    candidate_residue_indices: List[int] = []
    candidate_atom14_indices: List[int] = []
    topology_atom_indices: List[int] = []
    topology_residue_indices: List[int] = []
    residue_all_atom_indices: List[np.ndarray] = []
    residue_mapped_topology_indices: List[np.ndarray] = []
    residue_mapped_atom14_indices: List[np.ndarray] = []
    mapped_ca_topology: List[int] = []
    mapped_ca_candidate: List[int] = []
    active_residues = np.flatnonzero(candidate.node_mask)
    expected_atom_count = 0

    for candidate_index in range(candidate.n_residues):
        if not candidate.node_mask[candidate_index]:
            residue_all_atom_indices.append(np.empty((0,), dtype=np.int64))
            residue_mapped_topology_indices.append(np.empty((0,), dtype=np.int64))
            residue_mapped_atom14_indices.append(np.empty((0,), dtype=np.int64))
            continue
        token = _residue_token(candidate.residue_keys[candidate_index], ignore_chain)
        topology_residue_index = topology_index_by_token.get(token)
        if topology_residue_index is None:
            residue_all_atom_indices.append(np.empty((0,), dtype=np.int64))
            residue_mapped_topology_indices.append(np.empty((0,), dtype=np.int64))
            residue_mapped_atom14_indices.append(np.empty((0,), dtype=np.int64))
            continue

        expected_names = _expected_atom_names(int(candidate.aatype[candidate_index]))
        expected_residue_name = restype_1to3[restypes[int(candidate.aatype[candidate_index])]]
        observed_residue_name = standard_aliases.get(
            topology_residue_names[topology_residue_index],
            topology_residue_names[topology_residue_index],
        )
        if observed_residue_name != expected_residue_name:
            raise ValueError(
                f"Residue identity mismatch for {candidate.residue_keys[candidate_index]}: "
                f"candidate={expected_residue_name}, OpenMM={observed_residue_name}"
            )
        topology_atoms = topology_atoms_by_residue[topology_residue_index]
        atom_index_by_name = {atom.atom_name: atom.index for atom in topology_atoms}
        if len(atom_index_by_name) != len(topology_atoms):
            raise ValueError(
                f"OpenMM residue {topology_residue_index} has duplicate atom names"
            )
        mapped_topology: List[int] = []
        mapped_atom14: List[int] = []
        endpoint_valid = candidate.atom14_mask[[0, -1], candidate_index].all(axis=0)
        for atom14_index, atom_name in enumerate(expected_names):
            if not atom_name or not endpoint_valid[atom14_index]:
                continue
            expected_atom_count += 1
            topology_atom_index = atom_index_by_name.get(atom_name)
            if topology_atom_index is None:
                continue
            candidate_residue_indices.append(candidate_index)
            candidate_atom14_indices.append(atom14_index)
            topology_atom_indices.append(topology_atom_index)
            topology_residue_indices.append(topology_residue_index)
            mapped_topology.append(topology_atom_index)
            mapped_atom14.append(atom14_index)
            if atom_name == "CA":
                mapped_ca_topology.append(topology_atom_index)
                mapped_ca_candidate.append(candidate_index)

        required_backbone = {"N", "CA", "C"}
        mapped_names = {expected_names[index] for index in mapped_atom14}
        if not required_backbone.issubset(mapped_names):
            raise ValueError(
                f"Incomplete OpenMM backbone mapping for {candidate.residue_keys[candidate_index]}: "
                f"mapped={sorted(mapped_names)}"
            )
        residue_all_atom_indices.append(
            np.asarray([atom.index for atom in topology_atoms], dtype=np.int64)
        )
        residue_mapped_topology_indices.append(
            np.asarray(mapped_topology, dtype=np.int64)
        )
        residue_mapped_atom14_indices.append(
            np.asarray(mapped_atom14, dtype=np.int64)
        )

    mapped_residue_count = sum(
        mapped.size > 0 for mapped in residue_mapped_topology_indices
    )
    residue_fraction = mapped_residue_count / max(int(active_residues.size), 1)
    atom_fraction = len(topology_atom_indices) / max(expected_atom_count, 1)
    if residue_fraction < minimum_residue_fraction:
        raise ValueError(
            f"OpenMM residue mapping coverage {residue_fraction:.4f} is below "
            f"{minimum_residue_fraction:.4f}"
        )
    if atom_fraction < minimum_atom_fraction:
        raise ValueError(
            f"OpenMM heavy-atom mapping coverage {atom_fraction:.4f} is below "
            f"{minimum_atom_fraction:.4f}"
        )
    if len(mapped_ca_topology) < 3:
        raise ValueError("At least three mapped CA atoms are required for alignment")

    return CandidateTopologyMapping(
        candidate_residue_indices=np.asarray(candidate_residue_indices, dtype=np.int64),
        candidate_atom14_indices=np.asarray(candidate_atom14_indices, dtype=np.int64),
        topology_atom_indices=np.asarray(topology_atom_indices, dtype=np.int64),
        topology_residue_indices=np.asarray(topology_residue_indices, dtype=np.int64),
        residue_all_atom_indices=tuple(residue_all_atom_indices),
        residue_mapped_topology_indices=tuple(residue_mapped_topology_indices),
        residue_mapped_atom14_indices=tuple(residue_mapped_atom14_indices),
        mapped_ca_topology_indices=np.asarray(mapped_ca_topology, dtype=np.int64),
        mapped_ca_candidate_residue_indices=np.asarray(
            mapped_ca_candidate, dtype=np.int64
        ),
        mapped_residue_fraction=float(residue_fraction),
        mapped_atom_fraction=float(atom_fraction),
        ignored_chain_labels=bool(ignore_chain),
    )


def kabsch_row_transform(
    source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must have equal [N, 3] shapes")
    if source.shape[0] < 3:
        raise ValueError("Kabsch alignment requires at least three points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_center - source_center @ rotation
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Kabsch alignment produced non-finite values")
    return rotation, translation


def apply_row_transform(
    coordinates: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return np.asarray(coordinates, dtype=np.float64) @ rotation + translation


def align_candidate_to_topology(
    candidate: PathCandidate,
    mapping: CandidateTopologyMapping,
    topology_positions_angstrom: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    topology_positions = np.asarray(topology_positions_angstrom, dtype=np.float64)
    source_ca = candidate.atom14_pos_angstrom[
        -1, mapping.mapped_ca_candidate_residue_indices, 1
    ]
    target_ca = topology_positions[mapping.mapped_ca_topology_indices]
    rotation, translation = kabsch_row_transform(source_ca, target_ca)
    aligned = apply_row_transform(
        candidate.atom14_pos_angstrom.reshape(-1, 3), rotation, translation
    ).reshape(candidate.atom14_pos_angstrom.shape)
    aligned_ca = aligned[-1, mapping.mapped_ca_candidate_residue_indices, 1]
    ca_error = np.linalg.norm(aligned_ca - target_ca, axis=-1)
    diagnostics = {
        "holo_ca_alignment_rms_angstrom": float(
            np.sqrt(np.mean(ca_error * ca_error))
        ),
        "holo_ca_alignment_max_angstrom": float(ca_error.max()),
        "alignment_rotation_det": float(np.linalg.det(rotation)),
    }
    return aligned, diagnostics


def inject_candidate_frame(
    current_positions_angstrom: np.ndarray,
    target_atom14_angstrom: np.ndarray,
    mapping: CandidateTopologyMapping,
) -> np.ndarray:
    """Move mapped residues coherently, then set mapped heavy atoms exactly."""
    positions = np.asarray(current_positions_angstrom, dtype=np.float64).copy()
    target = np.asarray(target_atom14_angstrom, dtype=np.float64)
    if target.ndim != 3 or target.shape[-2:] != (14, 3):
        raise ValueError("target_atom14_angstrom must have shape [N, 14, 3]")
    for residue_index, (
        all_indices,
        mapped_topology,
        mapped_atom14,
    ) in enumerate(
        zip(
            mapping.residue_all_atom_indices,
            mapping.residue_mapped_topology_indices,
            mapping.residue_mapped_atom14_indices,
        )
    ):
        if mapped_topology.size == 0:
            continue
        name_to_pair = {
            int(atom14_index): int(topology_index)
            for atom14_index, topology_index in zip(mapped_atom14, mapped_topology)
        }
        anchor_atom14 = [index for index in (0, 1, 2) if index in name_to_pair]
        if len(anchor_atom14) != 3:
            raise ValueError(f"Residue {residue_index} has no complete N/CA/C mapping")
        anchor_topology = np.asarray(
            [name_to_pair[index] for index in anchor_atom14], dtype=np.int64
        )
        rotation, translation = kabsch_row_transform(
            positions[anchor_topology], target[residue_index, anchor_atom14]
        )
        positions[all_indices] = apply_row_transform(
            positions[all_indices], rotation, translation
        )
        positions[mapped_topology] = target[residue_index, mapped_atom14]
    if not np.isfinite(positions).all():
        raise ValueError("Candidate frame injection produced non-finite coordinates")
    return positions


def mapped_target_indices(
    mapping: CandidateTopologyMapping, restraint_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if restraint_mode not in {"ca", "backbone", "heavy"}:
        raise ValueError("restraint_mode must be 'ca', 'backbone', or 'heavy'")
    keep_atom14 = {
        "ca": {1},
        "backbone": {0, 1, 2},
        "heavy": set(range(14)),
    }[restraint_mode]
    keep = np.asarray(
        [index in keep_atom14 for index in mapping.candidate_atom14_indices],
        dtype=np.bool_,
    )
    if not keep.any():
        raise ValueError(f"No mapped atoms are available for restraint_mode={restraint_mode}")
    return (
        mapping.topology_atom_indices[keep],
        mapping.candidate_residue_indices[keep],
        mapping.candidate_atom14_indices[keep],
    )


def summarize_energy_profile(
    times: Sequence[float],
    energies_kj_mol: Sequence[float],
    *,
    interior_valid_mask: Sequence[bool] | None = None,
) -> Dict[str, Any]:
    times_array = np.asarray(times, dtype=np.float64)
    energies = np.asarray(energies_kj_mol, dtype=np.float64)
    if times_array.shape != energies.shape or times_array.ndim != 1:
        raise ValueError("Energy profile times and values must have equal 1D shapes")
    if not np.isfinite(energies).all():
        raise ValueError("Energy profile contains non-finite values")
    endpoint_baseline = (
        (1.0 - times_array) * energies[0] + times_array * energies[-1]
    )
    excess = energies - endpoint_baseline
    all_interior = excess[1:-1]
    if all_interior.size == 0:
        raise ValueError("Energy profile has no interior frames")
    if interior_valid_mask is None:
        valid = np.ones(times_array.shape, dtype=np.bool_)
    else:
        valid = np.asarray(interior_valid_mask, dtype=np.bool_)
        if valid.shape != times_array.shape:
            raise ValueError("Energy-profile validity mask does not match the time grid")
    interior = all_interior[valid[1:-1]]
    result: Dict[str, Any] = {
        "endpoint_apo_energy_kj_mol": float(energies[0]),
        "endpoint_holo_energy_kj_mol": float(energies[-1]),
        "interior_frames_total": int(all_interior.size),
        "interior_frames_included": int(interior.size),
        "interior_frames_excluded": int(all_interior.size - interior.size),
    }
    if interior.size == 0:
        result.update(
            {
                "interior_excess_mean_kj_mol": None,
                "interior_excess_p95_kj_mol": None,
                "interior_excess_max_kj_mol": None,
                "interior_positive_excess_mean_kj_mol": None,
                "interior_positive_excess_p95_kj_mol": None,
                "interior_positive_excess_max_kj_mol": None,
            }
        )
        return result
    positive = np.maximum(interior, 0.0)
    result.update(
        {
        "interior_excess_mean_kj_mol": float(interior.mean()),
        "interior_excess_p95_kj_mol": float(np.quantile(interior, 0.95)),
        "interior_excess_max_kj_mol": float(interior.max()),
        "interior_positive_excess_mean_kj_mol": float(positive.mean()),
        "interior_positive_excess_p95_kj_mol": float(np.quantile(positive, 0.95)),
        "interior_positive_excess_max_kj_mol": float(positive.max()),
        }
    )
    return result


def validate_openmm_gate0_score_report(
    report: Mapping[str, Any],
    *,
    label: str = "OpenMM Gate-0 score report",
    require_reference_cache: bool = False,
) -> None:
    if report.get("schema_version") != OPENMM_GATE0_SCORE_SCHEMA_VERSION:
        raise ValueError(
            f"{label} has unsupported schema {report.get('schema_version')!r}; "
            f"expected {OPENMM_GATE0_SCORE_SCHEMA_VERSION!r}"
        )
    if report.get("status") != "completed":
        raise ValueError(f"{label} is not completed")
    if not str(report.get("sample_id") or ""):
        raise ValueError(f"{label} misses sample_id")

    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label} misses its scorer contract")
    if require_reference_cache:
        if contract.get("frame_initialization") != "reference_cache":
            raise ValueError(f"{label} does not use the reference-cache contract")
        cache_sha = str(contract.get("frame_reference_cache_sha256") or "")
        if len(cache_sha) != 64:
            raise ValueError(f"{label} has an invalid frame-reference cache SHA-256")
        try:
            int(cache_sha, 16)
        except ValueError as exc:
            raise ValueError(
                f"{label} has a non-hexadecimal frame-reference cache SHA-256"
            ) from exc
    validity_contract = contract.get("relaxed_frame_validity")
    if not isinstance(validity_contract, Mapping):
        raise ValueError(f"{label} misses its relaxed-frame validity contract")
    threshold = float(validity_contract.get("maximum_allowed_kj_mol_nm", math.nan))
    expected_contract = relaxed_frame_validity_contract(threshold)
    if dict(validity_contract) != expected_contract:
        raise ValueError(f"{label} has an invalid relaxed-frame validity contract")

    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) < 3:
        raise ValueError(f"{label} requires at least three frame records")
    observed_times: List[float] = []
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{label} frame {expected_index} is not an object")
        if int(frame.get("frame_index", -1)) != expected_index:
            raise ValueError(f"{label} frame indices are not contiguous")
        time_value = float(frame.get("time", math.nan))
        if not math.isfinite(time_value):
            raise ValueError(f"{label} frame {expected_index} has a non-finite time")
        observed_times.append(time_value)
        for field in (
            "raw_potential_kj_mol",
            "relaxed_potential_kj_mol",
            RELAXED_VALIDITY_METRIC,
        ):
            value = float(frame.get(field, math.nan))
            if not math.isfinite(value):
                raise ValueError(
                    f"{label} frame {expected_index} has non-finite {field}"
                )
        observed_validity = frame.get("relaxed_validity")
        if not isinstance(observed_validity, Mapping):
            raise ValueError(
                f"{label} frame {expected_index} misses relaxed_validity"
            )
        expected_validity = assess_relaxed_frame_validity(
            float(frame[RELAXED_VALIDITY_METRIC]),
            maximum_residue_net_force_kj_mol_nm=threshold,
        )
        if dict(observed_validity) != expected_validity:
            raise ValueError(
                f"{label} frame {expected_index} relaxed validity is inconsistent"
            )
        if frame.get("relaxed_valid") is not expected_validity["valid"]:
            raise ValueError(
                f"{label} frame {expected_index} relaxed_valid is inconsistent"
            )
        if frame.get("relaxed_invalid_reasons") != expected_validity["invalid_reasons"]:
            raise ValueError(
                f"{label} frame {expected_index} invalid reasons are inconsistent"
            )
        minimization = frame.get("minimization")
        if not isinstance(minimization, Mapping):
            raise ValueError(f"{label} frame {expected_index} misses minimization audit")
        if not isinstance(minimization.get("reporter_available"), bool):
            raise ValueError(
                f"{label} frame {expected_index} has invalid reporter availability"
            )
        reporter_callback_count = minimization.get("reporter_callback_count")
        if reporter_callback_count is not None and int(reporter_callback_count) < 0:
            raise ValueError(
                f"{label} frame {expected_index} has invalid reporter callback count"
            )
        last_iteration = minimization.get("last_reported_iteration_index")
        if last_iteration is not None and int(last_iteration) < 0:
            raise ValueError(
                f"{label} frame {expected_index} has invalid minimization iteration index"
            )
        if not str(minimization.get("termination") or ""):
            raise ValueError(
                f"{label} frame {expected_index} misses minimization termination"
            )
        if not isinstance(
            minimization.get("termination_reason_available"), bool
        ):
            raise ValueError(
                f"{label} frame {expected_index} misses termination reason status"
            )
        maximum_iterations = int(minimization.get("maximum_iterations", 0))
        if maximum_iterations <= 0:
            raise ValueError(
                f"{label} frame {expected_index} has invalid maximum iterations"
            )
        for field in (
            "tolerance_kj_mol_nm",
            "unrestrained_potential_change_kj_mol",
        ):
            if not math.isfinite(float(minimization.get(field, math.nan))):
                raise ValueError(
                    f"{label} frame {expected_index} has invalid minimization/{field}"
                )

    times = np.asarray(observed_times, dtype=np.float64)
    if not np.isclose(times[0], 0.0, atol=1e-7) or not np.isclose(
        times[-1], 1.0, atol=1e-7
    ):
        raise ValueError(f"{label} frame grid does not have exact endpoints")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{label} frame times are not strictly increasing")

    relaxed_path = report.get("relaxed_path")
    if not isinstance(relaxed_path, Mapping):
        raise ValueError(f"{label} misses relaxed_path")
    interior = frames[1:-1]
    invalid_interior = sum(not bool(frame["relaxed_valid"]) for frame in interior)
    expected_counts = {
        "interior_frames": len(interior),
        "valid_interior_frames": len(interior) - invalid_interior,
        "invalid_interior_frames": invalid_interior,
    }
    for field, expected in expected_counts.items():
        if int(relaxed_path.get(field, -1)) != expected:
            raise ValueError(f"{label} has inconsistent relaxed_path/{field}")
    observed_reason_counts = relaxed_path.get("invalid_reason_counts")
    expected_interior_reasons: Dict[str, int] = {}
    for frame in interior:
        for reason in frame["relaxed_invalid_reasons"]:
            expected_interior_reasons[reason] = (
                expected_interior_reasons.get(reason, 0) + 1
            )
    if observed_reason_counts != expected_interior_reasons:
        raise ValueError(f"{label} has inconsistent invalid_reason_counts")

    raw_profile = report.get("raw_energy_profile")
    relaxed_profile = report.get("relaxed_energy_profile")
    if not isinstance(raw_profile, Mapping) or not isinstance(
        relaxed_profile, Mapping
    ):
        raise ValueError(f"{label} misses an energy profile")
    profile_expectations = (
        (raw_profile, len(interior)),
        (relaxed_profile, len(interior) - invalid_interior),
    )
    for profile, included in profile_expectations:
        if int(profile.get("interior_frames_total", -1)) != len(interior):
            raise ValueError(f"{label} has inconsistent energy-profile frame count")
        if int(profile.get("interior_frames_included", -1)) != included:
            raise ValueError(f"{label} has inconsistent included energy frames")
        if int(profile.get("interior_frames_excluded", -1)) != len(interior) - included:
            raise ValueError(f"{label} has inconsistent excluded energy frames")
    endpoint_fields = (
        ("endpoint_apo_energy_kj_mol", 0),
        ("endpoint_holo_energy_kj_mol", len(frames) - 1),
    )
    for profile, energy_field in (
        (raw_profile, "raw_potential_kj_mol"),
        (relaxed_profile, "relaxed_potential_kj_mol"),
    ):
        for profile_field, frame_index in endpoint_fields:
            if not math.isclose(
                float(profile.get(profile_field, math.nan)),
                float(frames[frame_index][energy_field]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(f"{label} has inconsistent endpoint energy")

    invalid_fraction = invalid_interior / len(interior)
    if not math.isclose(
        float(relaxed_path.get("invalid_frame_fraction", math.nan)),
        invalid_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} has inconsistent invalid_frame_fraction")
    severe_counts = np.asarray(
        [
            int(frame.get("protein_ligand_severe_clash_pairs", -1))
            + int(frame.get("protein_internal_severe_clash_pairs", -1))
            for frame in interior
        ],
        dtype=np.int64,
    )
    if np.any(severe_counts < 0):
        raise ValueError(f"{label} frame misses severe-clash counts")
    invalid_mask = np.asarray(
        [not bool(frame["relaxed_valid"]) for frame in interior], dtype=np.bool_
    )
    expected_path_fractions = {
        "severe_clash_frame_fraction": float(np.mean(severe_counts > 0)),
        "invalid_or_severe_clash_frame_fraction": float(
            np.mean(invalid_mask | (severe_counts > 0))
        ),
    }
    for field, expected in expected_path_fractions.items():
        if not math.isclose(
            float(relaxed_path.get(field, math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} has inconsistent relaxed_path/{field}")
    expected_severe_max = int(severe_counts.max())
    if int(relaxed_path.get("severe_clash_pairs_max", -1)) != expected_severe_max:
        raise ValueError(f"{label} has inconsistent severe_clash_pairs_max")


def validate_reference_topology_state(
    potential_kj_mol: float,
    atomic_forces_kj_mol_nm: np.ndarray,
    protein_atom_count: int,
    *,
    maximum_atomic_force_kj_mol_nm: float,
    state_label: str = "Prepared OpenMM reference topology",
) -> Dict[str, float]:
    """Reject an OpenMM reference state whose force field state is pathological."""
    forces = np.asarray(atomic_forces_kj_mol_nm, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3 or forces.shape[0] == 0:
        raise ValueError("atomic_forces_kj_mol_nm must have shape [A, 3]")
    if not 0 < int(protein_atom_count) <= forces.shape[0]:
        raise ValueError("protein_atom_count is outside the OpenMM atom axis")
    if maximum_atomic_force_kj_mol_nm <= 0.0:
        raise ValueError("maximum_atomic_force_kj_mol_nm must be positive")
    finite_energy = bool(np.isfinite(float(potential_kj_mol)))
    finite_forces = bool(np.isfinite(forces).all())
    base_diagnostics: Dict[str, Any] = {
        "potential_kj_mol": float(potential_kj_mol),
        "potential_is_finite": finite_energy,
        "atomic_forces_are_finite": finite_forces,
        "maximum_allowed_atomic_force_kj_mol_nm": float(
            maximum_atomic_force_kj_mol_nm
        ),
    }
    if not finite_energy or not finite_forces:
        raise ReferenceTopologyStateError(
            f"{state_label} is non-finite", base_diagnostics
        )

    norms = np.linalg.norm(forces, axis=-1)
    maximum = float(norms.max())
    diagnostics = {
        **base_diagnostics,
        "atomic_force_max_kj_mol_nm": maximum,
        "protein_atomic_force_max_kj_mol_nm": float(
            norms[: int(protein_atom_count)].max()
        ),
    }
    if maximum > float(maximum_atomic_force_kj_mol_nm):
        raise ReferenceTopologyStateError(
            f"{state_label} failed the physical preflight: "
            f"atomic_force_max={maximum:.6g} kJ/mol/nm exceeds "
            f"{float(maximum_atomic_force_kj_mol_nm):.6g}",
            diagnostics,
        )
    return diagnostics


def rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("RMS distance inputs must have equal [N, 3] shapes")
    if left.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=-1))))
