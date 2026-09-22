import numpy as np
from ase.io import read
from ase import Atoms

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "surface_tall.xyz"
OUTPUT_FILE = "surface_tall_H_removed.xyz"

REMOVE_PERCENT = 20.0
RANDOM_SEED = 42

BUFFER = 0.1  # Angstrom on each side


# ============================================================
# READ STRUCTURE
# ============================================================

atoms = read(INPUT_FILE)

symbols = np.array(atoms.get_chemical_symbols())
h_indices = np.where(symbols == "H")[0]

n_h_initial = len(h_indices)

print(f"Initial atoms : {len(atoms)}")
print(f"Initial H     : {n_h_initial}")


# ============================================================
# RANDOMLY REMOVE HYDROGENS
# ============================================================

if not 0.0 <= REMOVE_PERCENT <= 100.0:
    raise ValueError("REMOVE_PERCENT must be between 0 and 100.")

n_remove = int(round(n_h_initial * REMOVE_PERCENT / 100.0))

rng = np.random.default_rng(RANDOM_SEED)

remove_indices = rng.choice(
    h_indices,
    size=n_remove,
    replace=False
)

keep_mask = np.ones(len(atoms), dtype=bool)
keep_mask[remove_indices] = False

new_atoms = atoms[keep_mask]


# ============================================================
# GET ACTUAL COORDINATE EXTENTS
# ============================================================

positions = new_atoms.get_positions()

mins = positions.min(axis=0)
maxs = positions.max(axis=0)

xmin, ymin, zmin = mins
xmax, ymax, zmax = maxs

print("\nOriginal coordinate bounds:")
print(f"X : {xmin:.8f} -> {xmax:.8f}")
print(f"Y : {ymin:.8f} -> {ymax:.8f}")
print(f"Z : {zmin:.8f} -> {zmax:.8f}")


# ============================================================
# DEFINE NEW LATTICE FROM COORDINATE RANGE
# ============================================================

Lx = (xmax - xmin) + 2.0 * BUFFER
Ly = (ymax - ymin) + 2.0 * BUFFER
Lz = (zmax - zmin) + 2.0 * BUFFER

print("\nNew lattice dimensions:")
print(f"Lx = {Lx:.8f} Å")
print(f"Ly = {Ly:.8f} Å")
print(f"Lz = {Lz:.8f} Å")


# ============================================================
# SHIFT ATOMS INTO NEW CELL
#
# smallest coordinate becomes 0.1 Å
# largest coordinate becomes L - 0.1 Å
# ============================================================

shift = np.array([
    BUFFER - xmin,
    BUFFER - ymin,
    BUFFER - zmin
])

new_positions = positions + shift

new_atoms.set_positions(new_positions)

new_atoms.set_cell([
    [Lx, 0.0, 0.0],
    [0.0, Ly, 0.0],
    [0.0, 0.0, Lz]
])

new_atoms.set_pbc([True, True, True])


# ============================================================
# VERIFY NEW BOUNDS
# ============================================================

shifted_positions = new_atoms.get_positions()

new_mins = shifted_positions.min(axis=0)
new_maxs = shifted_positions.max(axis=0)

print("\nShifted coordinate bounds:")
print(f"X : {new_mins[0]:.8f} -> {new_maxs[0]:.8f}")
print(f"Y : {new_mins[1]:.8f} -> {new_maxs[1]:.8f}")
print(f"Z : {new_mins[2]:.8f} -> {new_maxs[2]:.8f}")


# ============================================================
# WRITE XYZ MANUALLY
# ============================================================

with open(OUTPUT_FILE, "w") as f:

    # Number of atoms
    f.write(f"{len(new_atoms)}\n")

    # Comment line with actual calculated lattice
    f.write(
        f'Lattice="'
        f'{Lx:.8f} 0.00000000 0.00000000 '
        f'0.00000000 {Ly:.8f} 0.00000000 '
        f'0.00000000 0.00000000 {Lz:.8f}" '
        f'pbc="T T T"\n'
    )

    # Coordinates
    for atom in new_atoms:

        x, y, z = atom.position

        f.write(
            f"{atom.symbol:<2s} "
            f"{x:16.8f} "
            f"{y:16.8f} "
            f"{z:16.8f}\n"
        )


# ============================================================
# SUMMARY
# ============================================================

n_h_final = sum(atom.symbol == "H" for atom in new_atoms)

print("\nFinished")
print("--------------------------------")
print(f"H before    : {n_h_initial}")
print(f"H removed   : {n_remove}")
print(f"H remaining : {n_h_final}")
print(f"Total atoms : {len(new_atoms)}")
print(f"Buffer      : {BUFFER} Å")
print(f"Lattice     : {Lx:.8f} x {Ly:.8f} x {Lz:.8f} Å")
print(f"PBC         : T T T")
print(f"Output      : {OUTPUT_FILE}")