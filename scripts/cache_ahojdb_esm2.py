#!/usr/bin/env python3
"""
ESM-2 Cache for AHoJ-DB (Stage-2)
- Input: data directory containing samples/<sample_id>/apo.pdb
- Output: samples/<sample_id>/esm.pt
"""

import os
import sys
import platform
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
import argparse
from tqdm import tqdm

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
    
    def __init__(self, data_dir: str, use_fallback: bool = False, batch_size: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        
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
            
    def extract_sequence(self, pdb_path: Path) -> Optional[str]:
        if not pdb_path.exists():
            return None
        try:
            structure = self.parser.get_structure(pdb_path.stem, str(pdb_path))
            residues = [r for r in structure.get_residues() if is_aa(r, standard=True)]
            if not residues: return None
            
            seq = ""
            for res in residues:
                seq += AA_MAP.get(res.get_resname().strip(), 'X')
            return seq
        except Exception as e:
            # print(f"Seq extraction failed for {pdb_path}: {e}")
            return None

    def encode_sequence(self, sample_id: str, sequence: str) -> Optional[Dict]:
        try:
            data = [(sample_id, sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            
            token_reprs = results['representations'][self.model.num_layers]
            
            # Remove start/end tokens
            per_residue = token_reprs[0, 1 : len(sequence) + 1]
            sequence_repr = token_reprs[0, 0] # CLS token
            
            return {
                'per_residue': per_residue.cpu(),
                'sequence': sequence_repr.cpu(),
                'sequence_str': sequence,
                'n_residues': len(sequence)
            }
        except Exception as e:
            print(f"Encoding failed for {sample_id}: {e}")
            return None

    def prepare_samples(self) -> List[Tuple[str, str, Path]]:
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
            esm_path = d / "esm.pt"
            if esm_path.exists():
                continue
                
            apo_pdb = d / "apo.pdb"
            if not apo_pdb.exists():
                continue
                
            seq = self.extract_sequence(apo_pdb)
            if seq:
                tasks.append((sample_id, seq, esm_path))
                
        return tasks

    def run(self):
        tasks = self.prepare_samples()
        print(f"Found {len(tasks)} samples to process.")
        
        if not tasks:
            print("No samples to process.")
            return

        # Sort by length to minimize padding (smart batching)
        tasks.sort(key=lambda x: len(x[1]))
        
        # Batch processing
        batch_size = self.batch_size
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        print(f"Processing in {total_batches} batches (Batch Size: {batch_size})...")
        
        for i in tqdm(range(0, len(tasks), batch_size), desc="Inference"):
            batch_tasks = tasks[i : i + batch_size]
            
            # Prepare batch data for ESM
            # batch_converter input: list of (id, seq)
            batch_data = [(t[0], t[1]) for t in batch_tasks]
            
            try:
                _, _, batch_tokens = self.batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)
                
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                
                token_reprs = results['representations'][self.model.num_layers]
                
                # Iterate and save
                for j, (sample_id, seq, out_path) in enumerate(batch_tasks):
                    # Extract individual sample from batch
                    # Note: batch_tokens includes padding, we need to slice using true length
                    # ESM encoding adds <cls> at start and <eos> at end
                    # token_reprs shape: [Batch, MaxLen, Dim]
                    
                    seq_len = len(seq)
                    # Slicing: [1 : seq_len + 1] to skip <cls> and exclude <eos>/padding
                    per_residue = token_reprs[j, 1 : seq_len + 1].cpu()
                    sequence_repr = token_reprs[j, 0].cpu() # <cls> token
                    
                    encoding = {
                        'per_residue': per_residue,
                        'sequence': sequence_repr,
                        'sequence_str': seq,
                        'n_residues': seq_len
                    }
                    torch.save(encoding, out_path)
                    
            except Exception as e:
                print(f"Batch {i//batch_size} failed: {e}")
                # Fallback? Maybe skip this batch or retry one by one if critical
                continue
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    args = parser.parse_args()
    
    cacher = ESM2CacheAHoJ(args.data_dir, args.fallback, batch_size=args.batch_size)
    cacher.run()

if __name__ == "__main__":
    main()
