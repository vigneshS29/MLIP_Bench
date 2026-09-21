#!/usr/bin/env python3
"""
Batch convert all LAMMPS data files (.lmp) in a directory to XYZ with charges.
Each output is written into an "output_xyz" folder with the same filename but .xyz extension.
"""

import os
import glob

# --- Atomic mass → element dictionary (extend if needed) ---
mass_to_element = {
    1.008: "H",
    6.941: "Li",
    12.011: "C",
    14.007: "N",
    15.999: "O",
    18.998: "F",
    22.990: "Na",
    35.453: "Cl",
    39.948: "Ar",
    63.546: "Cu",
    75.000: "As"
}

def closest_element(mass, tol=0.2):
    """Find the closest element symbol for a given mass."""
    closest = min(mass_to_element, key=lambda x: abs(x - mass))
    if abs(closest - mass) > tol:
        print(f"⚠️  Warning: mass {mass} not close to known element, guessing {mass_to_element[closest]}")
    return mass_to_element[closest]

def read_masses(datafile, tol=0.2):
    """Parse Masses section of LAMMPS data file into {type_id: element}."""
    type_map = {}
    in_masses = False
    with open(datafile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("masses"):
                in_masses = True
                continue
            if in_masses:
                if line[0].isalpha():  # next section reached
                    break
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        type_id = int(parts[0])
                        mass = float(parts[1])
                    except ValueError:
                        continue
                    type_map[type_id] = closest_element(mass, tol)
    return type_map

def read_atoms_and_charges(datafile, type_map):
    """Parse Atoms section into [(id, element, x,y,z,q), ...]."""
    atoms = []
    in_atoms = False
    with open(datafile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if line[0].isalpha():  # next section reached
                    break
                parts = line.split()
                if len(parts) >= 7:
                    atom_id = int(parts[0])
                    type_id = int(parts[2])
                    q = float(parts[3])
                    x, y, z = map(float, parts[4:7])
                    sym = type_map.get(type_id, "X")
                    atoms.append((atom_id, sym, x, y, z, q))
    # sort by atom id
    return sorted(atoms, key=lambda x: x[0])

def lammps_data_to_xyz(data_file, output_xyz):
    """Convert single LAMMPS .lmp file to .xyz with charges."""
    type_map = read_masses(data_file)
    atoms = read_atoms_and_charges(data_file, type_map)

    with open(output_xyz, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated from LAMMPS data\n")
        for (_, sym, x, y, z, q) in atoms:
            f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f} {q: .6f}\n")

    print(f"✅ {os.path.basename(data_file)} → {os.path.basename(output_xyz)}")

def batch_convert(input_dir, output_dir="output_xyz"):
    """Convert all .lmp files in a directory to .xyz."""
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.lmp"))
    files += glob.glob(os.path.join(input_dir, "*.data"))
    if not files:
        print("⚠️  No .lmp or .data files found.")
        return
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(output_dir, base + ".xyz")
        lammps_data_to_xyz(f, out)

# --- Run ---
if __name__ == "__main__":
    # Change "." to your directory with .lmp files
    batch_convert("./")