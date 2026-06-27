"""Dataset for Stage-1-v2 teacher posterior student training."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.stage2.datasets.dataset_stage2 import ApoHoloBridgeDataset
from .schema import BOOL_FIELDS, FLOAT_FIELDS, TeacherPosteriorLabel, load_teacher_posterior_npz


@dataclass
class TeacherPosteriorBatch:
    esm: torch.Tensor
    torsion_apo: torch.Tensor
    node_mask: torch.Tensor
    N_apo: torch.Tensor
    Ca_apo: torch.Tensor
    C_apo: torch.Tensor
    lig_points: torch.Tensor
    lig_types: torch.Tensor
    lig_mask: torch.Tensor
    w_res: torch.Tensor
    aatype: torch.Tensor
    teacher_float: Dict[str, torch.Tensor]
    teacher_bool: Dict[str, torch.Tensor]
    teacher_source: List[str]
    teacher_label_paths: List[str]
    pdb_ids: List[str]
    n_residues: List[int]
    sequences: List[str]


def _safe_sample_id(sample_id: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def _load_manifest(label_dir: Path) -> Dict[str, Path]:
    manifest_path = label_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    mapping: Dict[str, Path] = {}
    for record in manifest.get("records", []):
        sample_id = record.get("sample_id")
        path = record.get("path")
        if not sample_id or not path:
            continue
        p = Path(path)
        if not p.is_absolute():
            p = label_dir / p
        mapping[str(sample_id)] = p
    return mapping


class TeacherPosteriorDataset(Dataset):
    """Stage-2 triplet sample plus matching teacher posterior label."""

    def __init__(
        self,
        data_dir: str,
        label_dir: str,
        split: str = "train",
        valid_samples_file: Optional[str] = None,
        max_lig_tokens: int = 128,
        require_label: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.base = ApoHoloBridgeDataset(
            data_dir=data_dir,
            split=split,
            valid_samples_file=valid_samples_file,
            max_lig_tokens=max_lig_tokens,
        )
        self.manifest_map = _load_manifest(self.label_dir)
        self.require_label = bool(require_label)
        self.samples = []
        for idx, sample in enumerate(self.base.samples):
            sample_id = sample.get("id", f"sample_{idx}")
            label_path = self._resolve_label_path(sample_id)
            if label_path.exists():
                self.samples.append((idx, sample_id, label_path))
            elif self.require_label:
                continue
            else:
                self.samples.append((idx, sample_id, label_path))
        if not self.samples:
            raise ValueError(f"No teacher posterior labels found under {self.label_dir}")
        print(f"✓ Stage-1-v2 {split} teacher labels: {len(self.samples)} / {len(self.base)}")

    def _resolve_label_path(self, sample_id: str) -> Path:
        if sample_id in self.manifest_map:
            return self.manifest_map[sample_id]
        return self.label_dir / f"{_safe_sample_id(sample_id)}.npz"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        base_idx, sample_id, label_path = self.samples[idx]
        sample = self.base[base_idx]
        label = load_teacher_posterior_npz(label_path, expected_sample_id=sample_id)
        n_res = int(sample["n_residues"])
        if label.n_residues != n_res:
            raise ValueError(f"{sample_id} label n_residues={label.n_residues}, sample n_residues={n_res}")
        return {
            **sample,
            "teacher_label": label,
            "teacher_label_path": str(label_path),
        }


def _pad_array(batch_size: int, max_len: int, arrays: List[np.ndarray], shape_tail, dtype):
    out = np.zeros((batch_size, max_len, *shape_tail), dtype=dtype)
    for i, arr in enumerate(arrays):
        n = arr.shape[0]
        out[i, :n] = arr
    return torch.from_numpy(out)


def collate_teacher_posterior_batch(samples: List[Dict]) -> TeacherPosteriorBatch:
    batch_size = len(samples)
    max_n_res = max(int(s["n_residues"]) for s in samples)
    max_lig = max(len(s["lig_points"]) for s in samples)

    def pad_res(name: str, tail, dtype=np.float32):
        return _pad_array(batch_size, max_n_res, [np.asarray(s[name]) for s in samples], tail, dtype)

    esm = pad_res("esm", (1280,), np.float32)
    torsion_apo = pad_res("torsion_apo", (7,), np.float32)
    node_mask = np.zeros((batch_size, max_n_res), dtype=np.bool_)
    N_apo = pad_res("N_apo", (3,), np.float32)
    Ca_apo = pad_res("Ca_apo", (3,), np.float32)
    C_apo = pad_res("C_apo", (3,), np.float32)
    w_res = np.zeros((batch_size, max_n_res), dtype=np.float32)
    aatype = np.zeros((batch_size, max_n_res), dtype=np.int64)

    lig_points = np.zeros((batch_size, max_lig, 3), dtype=np.float32)
    lig_types = np.zeros((batch_size, max_lig, samples[0]["lig_types"].shape[-1]), dtype=np.float32)
    lig_mask = np.zeros((batch_size, max_lig), dtype=np.bool_)

    teacher_float = {field: np.zeros((batch_size, max_n_res), dtype=np.float32) for field in FLOAT_FIELDS}
    teacher_bool = {field: np.zeros((batch_size, max_n_res), dtype=np.bool_) for field in BOOL_FIELDS}

    pdb_ids: List[str] = []
    n_residues: List[int] = []
    sequences: List[str] = []
    teacher_source: List[str] = []
    teacher_label_paths: List[str] = []

    for i, sample in enumerate(samples):
        n_res = int(sample["n_residues"])
        n_lig = len(sample["lig_points"])
        label: TeacherPosteriorLabel = sample["teacher_label"]

        node_mask[i, :n_res] = sample["node_mask"]
        w_res[i, :n_res] = sample["w_res"]
        aatype[i, :n_res] = sample["aatype"]
        lig_points[i, :n_lig] = sample["lig_points"]
        lig_types[i, :n_lig] = sample["lig_types"]
        lig_mask[i, :n_lig] = True

        for field, arr in label.floats.items():
            teacher_float[field][i, :n_res] = arr
        for field, arr in label.bools.items():
            teacher_bool[field][i, :n_res] = arr

        pdb_ids.append(sample["id"])
        n_residues.append(n_res)
        sequences.append(sample.get("sequence", ""))
        teacher_source.append(label.teacher_source)
        teacher_label_paths.append(sample["teacher_label_path"])

    return TeacherPosteriorBatch(
        esm=esm,
        torsion_apo=torsion_apo,
        node_mask=torch.from_numpy(node_mask),
        N_apo=N_apo,
        Ca_apo=Ca_apo,
        C_apo=C_apo,
        lig_points=torch.from_numpy(lig_points),
        lig_types=torch.from_numpy(lig_types),
        lig_mask=torch.from_numpy(lig_mask),
        w_res=torch.from_numpy(w_res),
        aatype=torch.from_numpy(aatype),
        teacher_float={k: torch.from_numpy(v) for k, v in teacher_float.items()},
        teacher_bool={k: torch.from_numpy(v) for k, v in teacher_bool.items()},
        teacher_source=teacher_source,
        teacher_label_paths=teacher_label_paths,
        pdb_ids=pdb_ids,
        n_residues=n_residues,
        sequences=sequences,
    )
