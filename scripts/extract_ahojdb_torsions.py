#!/usr/bin/env python3
"""
Extract Backbone and Sidechain Torsions for AHoJ-DB Triplets (Stage-2)
- Input: output directory of prepare_ahojdb_triplets.py (containing apo.pdb, holo.pdb)
- Output: torsion_apo.npz, torsion_holo.npz in the same directory
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import concurrent.futures
from tqdm import tqdm
import warnings

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.residue_identity import (
    RESIDUE_ALIGNMENT_VERSION,
    iter_standard_residues,
    residue_key_from_biopython,
    residue_keys_to_array,
    residue_names_to_sequence,
)

# Add project root to path to import potential utils if needed, 
# but here we keep it standalone similar to extract_torsions.py for robustness.

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral, Vector
    from Bio.PDB.Residue import Residue
except ImportError as e:
    print(f"BioPython ImportError: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Sidechain definitions (CHI angles) - Same as original script
CHI_ANGLES_ATOMS = {
    'ALA': [],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], 
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], 
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
}

class TorsionExtractor:
    def __init__(self):
        # Use PERMISSIVE=1 to handle duplicate residue IDs
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
        import warnings
        warnings.simplefilter('ignore', PDBConstructionWarning)
        self.parser = PDBParser(QUIET=True, PERMISSIVE=1)

    def get_atom_coord(self, residue, atom_name: str) -> Optional[np.ndarray]:
        """Get atom coordinate, handling disordered atoms safely"""
        if atom_name not in residue:
            return None
        
        try:
            atom = residue[atom_name]
            # For disordered atoms, just take the first one
            if hasattr(atom, 'get_coord'):
                coord = atom.get_coord()
            else:
                # If it's a DisorderedAtom, get any child
                coord = list(atom.child_dict.values())[0].get_coord()
            
            if coord is None:
                return None
            return np.array(coord, dtype=np.float32)
        except:
            return None

    def calc_dihedral_angle(self, residue, atom_names, prev_residue=None, next_residue=None):
        coords = []
        for name in atom_names:
            if name == '-C' and prev_residue:
                c = self.get_atom_coord(prev_residue, 'C')
            elif name == '+N' and next_residue:
                c = self.get_atom_coord(next_residue, 'N')
            else:
                c = self.get_atom_coord(residue, name)
            if c is None: return None
            coords.append(c)
        
        try:
            v1, v2, v3, v4 = [Vector(c) for c in coords]
            return float(calc_dihedral(v1, v2, v3, v4))
        except:
            return None

    def is_sequential(self, r1, r2):
        if not r1 or not r2: return False
        if r1.get_parent().id != r2.get_parent().id: return False
        c_prev = self.get_atom_coord(r1, 'C')
        n_next = self.get_atom_coord(r2, 'N')
        if c_prev is None or n_next is None:
            return False
        peptide_distance = float(np.linalg.norm(c_prev - n_next))
        return 0.8 <= peptide_distance <= 2.0

    def extract(self, pdb_path: Path) -> Optional[Dict]:
        if not pdb_path.exists():
            return None
        
        try:
            # Parse structure - catch all PDB parsing errors here
            structure = self.parser.get_structure(pdb_path.stem, str(pdb_path))
        except Exception as e:
            # Skip files with parsing errors (duplicate IDs, malformed structures, etc.)
            import sys
            print(f"Skipping {pdb_path.name}: {str(e)[:50]}", file=sys.stderr)
            return None
        
        try:
            residues = list(iter_standard_residues(structure))
            
            if not residues: return None

            residue_keys = [
                residue_key_from_biopython(residue.get_parent(), residue)
                for residue in residues
            ]
            residue_names = [residue.get_resname().strip().upper() for residue in residues]

            n_res = len(residues)
            phi = np.zeros(n_res, dtype=np.float32)
            psi = np.zeros(n_res, dtype=np.float32)
            omega = np.zeros(n_res, dtype=np.float32)
            bb_mask = np.zeros(n_res, dtype=bool)
            omega_cis_trans = np.zeros(n_res, dtype=np.int8)
            chi = np.zeros((n_res, 4), dtype=np.float32)
            chi_mask = np.zeros((n_res, 4), dtype=bool)

            # Backbone
            for i, res in enumerate(residues):
                prev = residues[i-1] if i > 0 else None
                nxt = residues[i+1] if i < n_res - 1 else None
                
                # Check continuity
                has_prev = self.is_sequential(prev, res)
                has_next = self.is_sequential(res, nxt)

                # Phi
                if has_prev:
                    angle = self.calc_dihedral_angle(res, ['-C', 'N', 'CA', 'C'], prev_residue=prev)
                    if angle is not None:
                        phi[i] = angle
                        bb_mask[i] = True
                
                # Psi
                if has_next:
                    angle = self.calc_dihedral_angle(res, ['N', 'CA', 'C', '+N'], next_residue=nxt)
                    if angle is not None:
                        psi[i] = angle
                        bb_mask[i] = True
                
                # Omega
                if has_next:
                    angle = self.calc_dihedral_angle(res, ['CA', 'C', '+N', '+CA'], next_residue=nxt)
                    if angle is not None:
                        omega[i] = angle
                        omega_cis_trans[i] = 1 if abs(angle) < np.pi/6 else 0

                # Sidechain
                resname = res.get_resname().strip()
                if resname in CHI_ANGLES_ATOMS:
                    defs = CHI_ANGLES_ATOMS[resname]
                    for j, atoms in enumerate(defs):
                        if j >= 4: break
                        ang = self.calc_dihedral_angle(res, atoms)
                        if ang is not None:
                            chi[i, j] = ang
                            chi_mask[i, j] = True
            
            return {
                'phi': phi, 'psi': psi, 'omega': omega,
                'chi': chi, 'bb_mask': bb_mask, 'chi_mask': chi_mask,
                'omega_cis_trans': omega_cis_trans,
                'n_residues': n_res,
                'residue_keys': residue_keys_to_array(residue_keys),
                'residue_names': np.asarray(residue_names, dtype=np.str_),
                'sequence_str': np.asarray(residue_names_to_sequence(residue_names)),
                'residue_alignment_version': np.asarray(RESIDUE_ALIGNMENT_VERSION),
            }

        except Exception as e:
            import sys
            print(f"Error processing {pdb_path}: {e}", file=sys.stderr)
            return None

def _torsion_cache_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if 'residue_keys' not in data or 'residue_alignment_version' not in data:
                return False
            return str(np.asarray(data['residue_alignment_version']).item()) == RESIDUE_ALIGNMENT_VERSION
    except Exception:
        return False


def _atomic_savez(path: Path, data: Dict) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, **data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
def process_sample(sample_dir: Path):
    """Process one sample - creates its own extractor to avoid threading issues"""
    sample_id = sample_dir.name
    
    # Create extractor for this thread
    extractor = TorsionExtractor()
    
    # Process Apo
    apo_pdb = sample_dir / "apo.pdb"
    apo_out = sample_dir / "torsion_apo.npz"
    if apo_pdb.exists() and not _torsion_cache_is_current(apo_out):
        data = extractor.extract(apo_pdb)
        if data:
            _atomic_savez(apo_out, data)
    
    # Process Holo
    holo_pdb = sample_dir / "holo.pdb"
    holo_out = sample_dir / "torsion_holo.npz"
    if holo_pdb.exists() and not _torsion_cache_is_current(holo_out):
        data = extractor.extract(holo_pdb)
        if data:
            _atomic_savez(holo_out, data)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Torsions for AHoJ-DB")
    parser.add_argument("--data-dir", required=True, help="Root directory containing sample subdirectories (output of prepare_ahojdb_triplets.py)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of threads")
    parser.add_argument("--sample-list", type=str, default=None,
                        help="Optional file containing sample IDs to process")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap after sample-list filtering; 0 means all")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples_dir = data_dir / "samples"
    if not samples_dir.exists():
        # Maybe the data_dir IS the samples dir? or direct structure
        # prepare_ahojdb_triplets outputs to output_dir/samples/<id>
        if (data_dir / "index.json").exists(): # Standard output structure
             samples_dir = data_dir / "samples"
        else:
             # Fallback: assume data_dir contains samples directly
             samples_dir = data_dir
    
    if not samples_dir.exists():
        print(f"Error: {samples_dir} does not exist.")
        sys.exit(1)
        
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if args.sample_list:
        sample_list_path = Path(args.sample_list)
        if not sample_list_path.is_absolute() and not sample_list_path.exists():
            sample_list_path = data_dir / sample_list_path
        if not sample_list_path.exists():
            raise FileNotFoundError(sample_list_path)
        requested = {
            line.strip()
            for line in sample_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        sample_dirs = [sample_dir for sample_dir in sample_dirs if sample_dir.name in requested]
    sample_dirs.sort(key=lambda path: path.name)
    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]
    print(f"Found {len(sample_dirs)} samples in {samples_dir}")
    
    # Run in parallel using ProcessPoolExecutor for true parallelism
    # ProcessPoolExecutor bypasses GIL and uses multiple CPU cores effectively
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        chunksize = max(1, min(16, len(sample_dirs) // max(args.num_workers * 8, 1)))
        list(tqdm(executor.map(process_sample, sample_dirs, chunksize=chunksize),
                  total=len(sample_dirs), desc="Extracting Torsions"))

if __name__ == "__main__":
    main()
