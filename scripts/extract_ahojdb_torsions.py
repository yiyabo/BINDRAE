#!/usr/bin/env python3
"""
Extract Backbone and Sidechain Torsions for AHoJ-DB Triplets (Stage-2)
- Input: output directory of prepare_ahojdb_triplets.py (containing apo.pdb, holo.pdb)
- Output: torsion_apo.npz, torsion_holo.npz in the same directory
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import concurrent.futures
from tqdm import tqdm
import warnings

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
        self.parser = PDBParser(QUIET=True)

    def get_atom_coord(self, residue, atom_name: str) -> Optional[np.ndarray]:
        if atom_name not in residue:
            return None
        atom = residue[atom_name]
        if atom.is_disordered():
            atom = atom.selected_child
        coord = atom.coord
        if coord is None: return None
        return np.array(coord, dtype=np.float32)

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
        return abs(r1.id[1] - r2.id[1]) == 1

    def extract(self, pdb_path: Path) -> Optional[Dict]:
        if not pdb_path.exists():
            return None
        
        try:
            structure = self.parser.get_structure(pdb_path.stem, str(pdb_path))
            
            # More robust residue filtering - handle altLoc and non-standard naming
            standard_aa = {
                'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
                'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
                'THR', 'VAL', 'TRP', 'TYR'
            }
            
            residues = []
            for r in structure.get_residues():
                # Get residue name and clean it
                resname = r.get_resname().strip()
                # Handle altLoc: AALA -> ALA, BALA -> ALA
                if len(resname) == 4 and resname[0] in 'AB' and resname[1:] in standard_aa:
                    resname = resname[1:]
                
                # Check if it's a standard amino acid
                if resname in standard_aa:
                    residues.append(r)
            
            if not residues: return None

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
                        bb_mask[i] = True # Either phi or psi is enough to say backbone is somewhat valid? usually we want both.
                
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
                'n_residues': n_res
            }

        except Exception as e:
            # print(f"Error processing {pdb_path}: {e}")
            return None

def process_sample(sample_dir: Path, extractor: TorsionExtractor):
    sample_id = sample_dir.name
    
    # Proces Apo
    apo_pdb = sample_dir / "apo.pdb"
    if apo_pdb.exists() and not (sample_dir / "torsion_apo.npz").exists():
        data = extractor.extract(apo_pdb)
        if data:
            np.savez_compressed(sample_dir / "torsion_apo.npz", **data)
    
    # Process Holo
    holo_pdb = sample_dir / "holo.pdb"
    if holo_pdb.exists() and not (sample_dir / "torsion_holo.npz").exists():
        data = extractor.extract(holo_pdb)
        if data:
            np.savez_compressed(sample_dir / "torsion_holo.npz", **data)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Torsions for AHoJ-DB")
    parser.add_argument("--data-dir", required=True, help="Root directory containing sample subdirectories (output of prepare_ahojdb_triplets.py)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of threads")
    
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
    print(f"Found {len(sample_dirs)} samples in {samples_dir}")
    
    extractor = TorsionExtractor()
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample, d, extractor): d for d in sample_dirs}
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(sample_dirs), desc="Extracting Torsions"):
            pass

if __name__ == "__main__":
    main()
