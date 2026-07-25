#!/usr/bin/env python3
"""
ESM-2 Cache for AHoJ-DB (Stage-2)
- Input: data directory containing samples/<sample_id>/apo.pdb
- Output: samples/<sample_id>/esm.pt
  - per_residue: [N, D]
  - per_residue_layers: [N, K, D] when --last-k-layers > 1
"""

import os
import sys
import platform
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
import argparse
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.residue_identity import (
    RESIDUE_ALIGNMENT_VERSION,
    iter_standard_residues,
    load_residue_keys,
    residue_key_from_biopython,
    residue_keys_to_array,
    residue_names_to_sequence,
)

warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError as e:
    print(f"Error importing BioPython: {e}")
    sys.exit(1)

# Three-to-One
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

try:
    import esm
except ImportError as e:
    print(f"Error importing ESM: {e}")
    print("Please install: pip install fair-esm")
    sys.exit(1)


class ESM2CacheAHoJ:
    """ESM-2 Cacher for AHoJ-DB"""
    
    # Model Config (Same as cache_esm2.py)
    MODEL_CONFIG = {
        'Darwin': {  # macOS
            'name': 'esm2_t6_8M_UR50D',
            'fallback': 'esm2_t12_35M_UR50D',
            'batch_size': 1,
            'description': 'MacOS - using lightweight model'
        },
        'Linux': {  # Server
            'name': 'esm2_t33_650M_UR50D',
            'fallback': 'esm2_t33_650M_UR50D',
            'batch_size': 4,
            'description': 'Linux - using standard 650M model'
        }
    }
    
    def __init__(
        self,
        data_dir: str,
        use_fallback: bool = False,
        batch_size: Optional[int] = None,
        last_k_layers: int = 1,
        sample_list: Optional[str] = None,
        max_samples: int = 0,
        force: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        self.last_k_layers = int(last_k_layers)
        if self.last_k_layers < 1:
            raise ValueError(f"last_k_layers must be >= 1, got {last_k_layers}")
        self.sample_ids = self._load_sample_ids(sample_list)
        self.max_samples = int(max_samples)
        self.force = bool(force)
        
        # Determine system
        self.system = platform.system()
        self.device = self._get_device()
        self.model_name, self.default_batch_size = self._select_model(use_fallback)
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        
        print(f"\n{'='*80}")
        print(f"ESM-2 Cache for AHoJ-DB")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Last-K layers: {self.last_k_layers}")
        print(f"Sample list: {sample_list or 'ALL'}")
        print(f"Max samples: {self.max_samples if self.max_samples > 0 else 'ALL'}")
        print(f"Force: {self.force}")
        print(f"Data Dir: {self.data_dir}")
        
        # Load Model
        self.model, self.alphabet = self._load_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.parser = PDBParser(QUIET=True)
        
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _select_model(self, use_fallback: bool) -> Tuple[str, int]:
        if self.system not in self.MODEL_CONFIG:
            config = self.MODEL_CONFIG['Linux']
        else:
            config = self.MODEL_CONFIG[self.system]
            
        model_name = config['fallback'] if use_fallback else config['name']
        batch_size = config['batch_size']
        return model_name, batch_size
    
    def _load_model(self):
        print(f"Loading model: {self.model_name}...")
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
            model = model.to(self.device)
            model.eval()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            return model, alphabet
        except Exception as e:
            print(f"Model load failed: {e}")
            sys.exit(1)

    def _load_sample_ids(self, sample_list: Optional[str]) -> Optional[set]:
        if not sample_list:
            return None
        path = Path(sample_list)
        if not path.is_absolute() and not path.exists():
            path = self.data_dir / sample_list
        with open(path, "r") as f:
            ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(ids)} sample ids from {path}")
        return ids

    def _esm_cache_satisfies_request(self, esm_path: Path) -> bool:
        if self.force or not esm_path.exists():
            return False
        if self.last_k_layers <= 1:
            return True
        try:
            data = torch.load(esm_path, map_location="cpu", weights_only=False)
            layers = data.get("per_residue_layers") if isinstance(data, dict) else None
            return layers is not None and int(layers.shape[1]) >= self.last_k_layers
        except Exception:
            return False
            
    def extract_sequence_and_keys(self, pdb_path: Path):
        if not pdb_path.exists():
            return None
        try:
            structure = self.parser.get_structure(pdb_path.stem, str(pdb_path))
            residues = list(iter_standard_residues(structure))
            if not residues: return None
            residue_names = [res.get_resname().strip().upper() for res in residues]
            residue_keys = [
                residue_key_from_biopython(res.get_parent(), res)
                for res in residues
            ]
            return residue_names_to_sequence(residue_names), residue_keys
        except Exception as e:
            # print(f"Seq extraction failed for {pdb_path}: {e}")
            return None

    def extract_sequence(self, pdb_path: Path) -> Optional[str]:
        metadata = self.extract_sequence_and_keys(pdb_path)
        return metadata[0] if metadata is not None else None

    def extract_sample_sequence_and_keys(self, sample_dir: Path):
        """Prefer the canonical Stage-2 torsion axis over all chains in a PDB."""
        torsion_path = sample_dir / "torsion_apo.npz"
        if torsion_path.exists():
            try:
                with np.load(torsion_path, allow_pickle=False) as data:
                    residue_keys = load_residue_keys(data)
                    sequence = (
                        str(np.asarray(data["sequence_str"]).item())
                        if "sequence_str" in data
                        else ""
                    )
                if residue_keys is not None and sequence:
                    if len(residue_keys) != len(sequence):
                        raise ValueError(
                            f"{torsion_path} residue key count {len(residue_keys)} "
                            f"does not match sequence length {len(sequence)}"
                        )
                    return sequence, residue_keys
            except Exception as exc:
                print(
                    f"Canonical torsion metadata failed for {sample_dir.name}: {exc}"
                )

        return self.extract_sequence_and_keys(sample_dir / "apo.pdb")

    def encode_sequence(self, sample_id: str, sequence: str, residue_keys=None) -> Optional[Dict]:
        try:
            data = [(sample_id, sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            final_layer = int(self.model.num_layers)
            first_layer = max(1, final_layer - self.last_k_layers + 1)
            repr_layers = list(range(first_layer, final_layer + 1))
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=repr_layers)
            
            token_reprs = results['representations'][final_layer]
            
            # Remove start/end tokens
            per_residue = token_reprs[0, 1 : len(sequence) + 1]
            sequence_repr = token_reprs[0, 0] # CLS token
            per_residue_layers = torch.stack(
                [
                    results['representations'][layer][0, 1 : len(sequence) + 1]
                    for layer in repr_layers
                ],
                dim=1,
            )
            
            encoding = {
                'per_residue': per_residue.cpu(),
                'sequence': sequence_repr.cpu(),
                'sequence_str': sequence,
                'n_residues': len(sequence)
            }
            if residue_keys is not None:
                if len(residue_keys) != len(sequence):
                    raise ValueError(
                        f"residue key count {len(residue_keys)} does not match sequence "
                        f"length {len(sequence)}"
                    )
                encoding['residue_keys'] = residue_keys_to_array(residue_keys)
                encoding['residue_alignment_version'] = RESIDUE_ALIGNMENT_VERSION
            if self.last_k_layers > 1:
                encoding['per_residue_layers'] = per_residue_layers.cpu()
                encoding['esm_layer_indices'] = torch.tensor(repr_layers, dtype=torch.long)
            return encoding
        except Exception as e:
            print(f"Encoding failed for {sample_id}: {e}")
            return None

    def prepare_samples(self) -> List[Tuple[str, str, list, Path]]:
        """Collect all valid samples that need processing."""
        if not self.samples_dir.exists():
            if (self.data_dir / "index.json").exists():
                 pass
            else:
                 self.samples_dir = self.data_dir
        
        sample_dirs = sorted([d for d in self.samples_dir.iterdir() if d.is_dir()])
        print(f"Scanning {len(sample_dirs)} samples...")
        
        tasks = []
        for d in tqdm(sample_dirs, desc="Scanning"):
            sample_id = d.name
            if self.sample_ids is not None and sample_id not in self.sample_ids:
                continue
            esm_path = d / "esm.pt"
            if self._esm_cache_satisfies_request(esm_path):
                continue
                
            if not (d / "apo.pdb").exists():
                continue
                
            metadata = self.extract_sample_sequence_and_keys(d)
            if metadata:
                seq, residue_keys = metadata
                tasks.append((sample_id, seq, residue_keys, esm_path))
                if self.max_samples > 0 and len(tasks) >= self.max_samples:
                    break
                
        return tasks

    def run(self):
        tasks = self.prepare_samples()
        print(f"Found {len(tasks)} samples to process.")
        
        if not tasks:
            print("No samples to process.")
            return

        # Sort by length to minimize padding
        tasks.sort(key=lambda x: len(x[1]))

        batch_size = self.batch_size
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        print(f"Processing in {total_batches} batches (Start BS: {batch_size})...")

        for i in tqdm(range(0, len(tasks), batch_size), desc="Inference"):
            batch_tasks = tasks[i : i + batch_size]
            self.process_batch_safe(batch_tasks)

    def process_batch_safe(self, batch_tasks: List[Tuple[str, str, list, Path]]):
        """Process a batch with OOM automatic recovery (recursive splitting)."""
        if not batch_tasks:
            return

        try:
            # Prepare batch data
            batch_data = [(t[0], t[1]) for t in batch_tasks]
            
            _, _, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            
            final_layer = int(self.model.num_layers)
            first_layer = max(1, final_layer - self.last_k_layers + 1)
            repr_layers = list(range(first_layer, final_layer + 1))
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=repr_layers)
            
            token_reprs = results['representations'][final_layer]
            
            # Save results
            for j, (sample_id, seq, residue_keys, out_path) in enumerate(batch_tasks):
                seq_len = len(seq)
                per_residue = token_reprs[j, 1 : seq_len + 1].cpu()
                sequence_repr = token_reprs[j, 0].cpu()
                per_residue_layers = torch.stack(
                    [
                        results['representations'][layer][j, 1 : seq_len + 1]
                        for layer in repr_layers
                    ],
                    dim=1,
                ).cpu()
                
                encoding = {
                    'per_residue': per_residue,
                    'sequence': sequence_repr,
                    'sequence_str': seq,
                    'n_residues': seq_len,
                    'residue_keys': residue_keys_to_array(residue_keys),
                    'residue_alignment_version': RESIDUE_ALIGNMENT_VERSION,
                }
                if self.last_k_layers > 1:
                    encoding['per_residue_layers'] = per_residue_layers
                    encoding['esm_layer_indices'] = torch.tensor(repr_layers, dtype=torch.long)
                torch.save(encoding, out_path)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                n = len(batch_tasks)
                if n <= 1:
                    print(f"❌ Error: Sample {batch_tasks[0][0]} (len={len(batch_tasks[0][1])}) caused OOM even with batch_size=1. Skipping.")
                    return
                    
                mid = n // 2
                print(f"⚠️  OOM with batch={n}. Retrying with split batches ({mid}, {n-mid})...")
                self.process_batch_safe(batch_tasks[:mid])
                self.process_batch_safe(batch_tasks[mid:])
            else:
                print(f"❌ Batch inference failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    parser.add_argument("--last-k-layers", type=int, default=1,
                        help="Save last K per-residue ESM layers; 1 preserves legacy esm.pt schema")
    parser.add_argument("--sample-list", type=str, default=None,
                        help="Optional file with sample ids to process")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of samples to process after filtering; 0 means all")
    parser.add_argument("--force", action="store_true",
                        help="Recompute selected esm.pt files even if they already satisfy the request")
    args = parser.parse_args()
    
    cacher = ESM2CacheAHoJ(
        args.data_dir,
        args.fallback,
        batch_size=args.batch_size,
        last_k_layers=args.last_k_layers,
        sample_list=args.sample_list,
        max_samples=args.max_samples,
        force=args.force,
    )
    cacher.run()

if __name__ == "__main__":
    main()
