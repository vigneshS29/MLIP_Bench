import re
import argparse
import numpy as np
from collections import defaultdict

def compute_density(masses_section, atom_lines, box_dims):
    """Calculates density for a cuboidal box: [x, y, z]"""
    amu_to_g = 1.66053906660e-24
    A3_to_cm3 = 1e-24
    type_to_mass = {int(line.strip().split()[0]): float(line.strip().split()[1]) for line in masses_section}
    atom_type_counts = {}
    for line in atom_lines:
        parts = line.strip().split()
        atom_type = int(parts[2])
        atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1
    total_mass_amu = sum(type_to_mass[t] * c for t, c in atom_type_counts.items())
    total_mass_g = total_mass_amu * amu_to_g
    volume_cm3 = np.prod(box_dims) * A3_to_cm3
    return total_mass_g / volume_cm3

def write_xyz(atom_lines, box_dims, output_xyz, masses_section):
    mass_to_element = {1.008: "H", 6.94: "Li", 35.5: "Cl", 12.01: "C", 14.01: "N", 15.999: "O", 18.998: "F", 75.00: "As"}
    atom_type_to_element = {}
    for line in masses_section:
        parts = line.split()
        atom_type, mass = int(parts[0]), float(parts[1])
        closest = min(mass_to_element.keys(), key=lambda m: abs(m - mass))
        atom_type_to_element[atom_type] = mass_to_element[closest]
    with open(output_xyz, 'w') as f:
        f.write(f"{len(atom_lines)}\nGenerated structure Box: {box_dims}\n")
        for line in atom_lines:
            parts = line.split()
            symbol = atom_type_to_element.get(int(parts[2]), "X")
            f.write(f"{symbol} {float(parts[4]):.6f} {float(parts[5]):.6f} {float(parts[6]):.6f}\n")

def parse_lammps_data(content):
    sections = defaultdict(list)
    current_section = None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("LAMMPS data file"): continue
        match = re.match(r'^(Masses|Pair Coeffs|Bond Coeffs|Angle Coeffs|Dihedral Coeffs|Improper Coeffs|Atoms|Bonds|Angles|Dihedrals|Impropers)$', stripped)
        if match:
            current_section = match.group()
            continue
        if current_section and re.match(r'^\d', stripped):
            sections[current_section].append(stripped)
    return sections

def renumber_lines(lines, id_offset, type_offset, atom_offset):
    new_lines = []
    for line in lines:
        parts = line.split()
        new_id, new_type = int(parts[0]) + id_offset, int(parts[1]) + type_offset
        new_parts = [str(new_id), str(new_type)] + [str(int(p) + atom_offset) for p in parts[2:]]
        new_lines.append(" ".join(new_parts))
    return new_lines

def random_rotate_atoms(atom_lines, masses_section):
    alpha, beta, gamma = np.random.uniform(0, 2*np.pi, 3)
    R = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]) @ \
        np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]) @ \
        np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    type_to_mass = {int(line.split()[0]): float(line.split()[1]) for line in masses_section}
    coords = np.array([[float(p.split()[4]), float(p.split()[5]), float(p.split()[6])] for p in atom_lines])
    masses = np.array([type_to_mass.get(int(p.split()[2]), 1.0) for p in atom_lines])
    com = np.average(coords, axis=0, weights=masses)
    rotated = []
    for i, line in enumerate(atom_lines):
        new_vec = R @ (coords[i] - com) + com
        parts = line.split()
        rotated.append(" ".join(parts[:4] + [f"{new_vec[0]:.6f}", f"{new_vec[1]:.6f}", f"{new_vec[2]:.6f}"] + parts[7:]))
    return rotated

def generate_random_position(box_dims, padding=3.0):
    return [np.random.uniform(padding, dim - padding) for dim in box_dims]

def merge_multiple_lammps_files(lmp_files, mol_counts, output_path, box_dims):
    cumulative = defaultdict(list)
    total_counts = {k: 0 for k in ['Atoms', 'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types']}
    offsets = {'atom': 0, 'bond': 0, 'angle': 0, 'dihedral': 0, 'improper': 0}
    type_offsets = {'atom': 0, 'bond': 0, 'angle': 0, 'dihedral': 0, 'improper': 0}
    mol_id = 1

    for lmp_file, mol_count in zip(lmp_files, mol_counts):
        with open(lmp_file, 'r') as f: parsed = parse_lammps_data(f.read())
        cumulative['Masses'].extend([f"{int(l.split()[0]) + type_offsets['atom']} {l.split()[1]}" for l in parsed.get('Masses', [])])
        
        for key, t_off in zip(['Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs'], type_offsets.values()):
            cumulative[key].extend([f"{int(l.split()[0]) + t_off} " + " ".join(l.split()[1:]) for l in parsed.get(key, [])])

        for _ in range(mol_count):
            pos = generate_random_position(box_dims)
            atoms = random_rotate_atoms([f"{int(p.split()[0]) + offsets['atom']} {mol_id} {int(p.split()[2]) + type_offsets['atom']} " + " ".join(p.split()[3:]) for p in parsed.get('Atoms', [])], parsed['Masses'])
            cumulative['Atoms'].extend([" ".join(p.split()[:4] + [f"{float(p.split()[4])+pos[0]:.6f}", f"{float(p.split()[5])+pos[1]:.6f}", f"{float(p.split()[6])+pos[2]:.6f}"] + p.split()[7:]) for p in atoms])
            cumulative['Bonds'].extend(renumber_lines(parsed.get('Bonds', []), offsets['bond'], type_offsets['bond'], offsets['atom']))
            cumulative['Angles'].extend(renumber_lines(parsed.get('Angles', []), offsets['angle'], type_offsets['angle'], offsets['atom']))
            cumulative['Dihedrals'].extend(renumber_lines(parsed.get('Dihedrals', []), offsets['dihedral'], type_offsets['dihedral'], offsets['atom']))
            cumulative['Impropers'].extend(renumber_lines(parsed.get('Impropers', []), offsets['improper'], type_offsets['improper'], offsets['atom']))
            
            for k in offsets: offsets[k] += len(parsed.get(k.capitalize() + 's', []))
            mol_id += 1

        for k in type_offsets: 
            section = 'Masses' if k == 'atom' else k.capitalize() + ' Coeffs'
            type_offsets[k] += len(parsed.get(section, []))
            total_counts[k + ' types'] = type_offsets[k]

    # Write merged LAMMPS file
    with open(output_path, 'w') as f:
        f.write(f"Merged LAMMPS data\n\n{len(cumulative['Atoms'])} atoms\n{len(cumulative['Bonds'])} bonds\n{len(cumulative['Angles'])} angles\n{len(cumulative['Dihedrals'])} dihedrals\n{len(cumulative['Impropers'])} impropers\n\n")
        f.write(f"{total_counts['atom types']} atom types\n{total_counts['bond types']} bond types\n{total_counts['angle types']} angle types\n{total_counts['dihedral types']} dihedral types\n{total_counts['improper types']} improper types\n\n")
        f.write(f"0.0 {box_dims[0]} xlo xhi\n0.0 {box_dims[1]} ylo yhi\n0.0 {box_dims[2]} zlo zhi\n\n")
        for s in ['Masses', 'Atoms', 'Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            if cumulative[s]: f.write(f"{s}\n\n" + "\n".join(cumulative[s]) + "\n\n")

    # Write settings file with CORRECT pair_coeff format
    with open('settings.lmp', 'w') as f:
        for section in ['Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs']:
            if cumulative[section]:
                label = '_'.join(section.split(' ')).lower()[:-1]
                for line in cumulative[section]:
                    parts = line.split()
                    if section == 'Pair Coeffs':
                        # Fix: Repeat type ID for I and J, then add epsilon sigma and cutoff
                        f.write(f"pair_coeff\t{parts[0]}\t{parts[0]}\t{parts[1]}\t{parts[2]}\t12.0\n")
                    else:
                        f.write(f"{label}\t{parts[0]}\t" + "\t".join(parts[1:]) + "\n")
                f.write("\n")

    print(f"\n📦 Box: {box_dims} Å | ⚖️ Density: {compute_density(cumulative['Masses'], cumulative['Atoms'], box_dims):.4f} g/cm³")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lmp', required=True, help='Space-separated .lmp files')
    parser.add_argument('-N', default='10', help='Space-separated molecule counts')
    parser.add_argument('-O', dest='output', default='data.lmp', help='Output file')
    # Added dest='length' back in so args.length works properly!
    parser.add_argument('-L', dest='length', default='30.0', help='Box dimensions (e.g., "30" or "25 25 75")')
    
    args = parser.parse_args()
    
    dv = [float(x) for x in args.length.split()]
    box_dims = dv * 3 if len(dv) == 1 else dv
    
    merge_multiple_lammps_files(
        args.lmp.split(), 
        [int(n) for n in args.N.split()], 
        args.output, 
        box_dims
    )