import ijson
import json
import glob
import os
from pymatgen.core import Structure
from decimal import Decimal

ALEXANDRIA_DIR = '/home/vsbat/work/Alexandria'
ALEX_PATTERN = os.path.join(ALEXANDRIA_DIR, "alexandria*.json")
output_dir = os.path.join(ALEXANDRIA_DIR, "filtered_structures")
os.makedirs(output_dir, exist_ok=True)

def convert_decimals(obj):
    if isinstance(obj, list):
        return [convert_decimals(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

def get_structure_by_mat_id(mat_id: str):
    for filename in sorted(glob.glob(ALEX_PATTERN)):
        print(f"\nProcessing file: {filename}")
        try:
            with open(filename, "r") as f:
                for entry in ijson.items(f, "entries.item"):
                    if entry.get("data", {}).get("mat_id", "unknown") == mat_id:
                        structure_dict = entry.get("structure")
                        if structure_dict:
                            try:
                                clean_structure_dict = convert_decimals(structure_dict)
                                print(f'Structure {entry.get("data").get("formula")} found')
                                return Structure.from_dict(clean_structure_dict)
                            except Exception as parse_err:
                                print(f"  Error reading structure: {parse_err}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def get_poscar_by_mat_id(mat_id, poscar_path=None):
    if poscar_path is None:
        poscar_path = mat_id + "_POSCAR"
    structure = get_structure_by_mat_id(mat_id)
    if structure:
        with open(poscar_path, "w") as poscar_file:
            poscar_file.write(structure.to(fmt="poscar"))