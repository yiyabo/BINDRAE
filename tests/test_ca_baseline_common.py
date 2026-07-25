from pathlib import Path

import numpy as np

from scripts.ca_baseline_common import canonical_ca_pair, write_ca_only_pdb


def _records(coords):
    return [
        {
            "chain": "A",
            "resseq": str(index + 1),
            "icode": "",
            "resname": "ALA",
            "xyz": np.asarray(coord, dtype=np.float64),
        }
        for index, coord in enumerate(coords)
    ]


def test_canonical_ca_pair_prefers_backbone_cache(tmp_path: Path):
    raw_apo = _records([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    raw_holo = _records([[30.0, 0.0, 0.0], [31.0, 0.0, 0.0]])
    apo = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    holo = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    np.savez(tmp_path / "apo_backbone.npz", Ca=apo)
    np.savez(tmp_path / "holo_backbone.npz", Ca=holo)
    loaded_apo, loaded_holo, source = canonical_ca_pair(
        tmp_path, raw_apo, raw_holo
    )
    np.testing.assert_array_equal(loaded_apo, apo)
    np.testing.assert_array_equal(loaded_holo, holo)
    assert source == "backbone_cache"


def test_write_ca_only_pdb_rounds_coordinates(tmp_path: Path):
    records = _records([[0.0, 0.0, 0.0]])
    output = tmp_path / "ca.pdb"
    write_ca_only_pdb(output, records, np.array([[1.2344, 2.3455, 3.4566]]))
    text = output.read_text()
    assert "   1.234   2.345   3.457" in text


def test_canonical_ca_pair_rejects_mismatched_endpoint_axes(tmp_path: Path):
    apo_records = _records([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    holo_records = _records([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    np.savez(
        tmp_path / "apo_backbone.npz",
        Ca=np.zeros((2, 3)),
        residue_keys=np.array(["A|1|", "A|2|"]),
    )
    np.savez(
        tmp_path / "holo_backbone.npz",
        Ca=np.zeros((2, 3)),
        residue_keys=np.array(["B|1|", "B|3|"]),
    )
    try:
        canonical_ca_pair(tmp_path, apo_records, holo_records)
    except ValueError as exc:
        assert "holo PDB records" in str(exc) or "residue axes differ" in str(exc)
    else:
        raise AssertionError("Expected mismatched endpoint residue axes to fail")
