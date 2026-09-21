"""2 ps implantation cycles, with sputter cleanup every 0.5 ps.

Removal is a geometric heuristic: z > initial_surface_top + clearance
and vz > 0. Formulas describe total removed composition, not molecules.
Run with: python run_implantation.py
"""
import csv
import json
from pathlib import Path
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

SURFACE_XYZ = "surface.xyz"
IMPLANT_SPECIES = "Xe"
ENERGY_EV = 1000.0
CYCLES = 1
CYCLE_TIME_PS = 2.0
CLEANUP_TIME_PS = 0.5  # Cleanup interval: 0.5, 1.0, 1.5, and 2.0 ps.
DT_FS = 0.5
TEMP_K = 300.0
FRICTION = 0.9  # Inverse ASE time units; preserves your value.
FIXED_CUTOFF_Z = 3.0
THERMO_CUTOFF_Z = 10.0
ION_HEIGHT_A = 0.5  # Preserves your value; places Xe very close to the surface.
SPUTTER_CLEARANCE_A = 5.0  # User-adjustable operational removal threshold.
WRITE_INTERVAL = 100
OUTPUT_DIR = "implantation_output"  # Must not already exist.
DEVICE = "cuda"
RANDOM_SEED = None


def steps_for(time_ps):
    value = time_ps * 1000.0 / DT_FS
    steps = int(round(value))
    if not np.isclose(value, steps, rtol=0, atol=1e-8):
        raise ValueError("Requested duration must be a whole number of MD steps.")
    return steps


def formula(atoms):
    return atoms.get_chemical_formula(mode="hill") if len(atoms) else "none"


def sputter_mask(atoms, removal_z):
    return (
        (atoms.positions[:, 2] > removal_z)
        & (atoms.get_velocities()[:, 2] > 0.0)
        & ~atoms.arrays["fixed_atom"].astype(bool)
    )


def remove_sputtered(atoms, removal_z):
    mask = sputter_mask(atoms, removal_z)
    removed = atoms[mask]
    kept = atoms[~mask]
    kept.set_constraint(FixAtoms(mask=kept.arrays["fixed_atom"].astype(bool)))
    kept.calc = atoms.calc
    return kept, removed


def save_frame(atoms, path, cycle, cycle_step, event):
    snapshot = atoms.copy()
    snapshot.wrap(pbc=[True, True, False])
    snapshot.info.update(
        cycle=cycle,
        cycle_time_ps=cycle_step * DT_FS / 1000.0,
        total_time_ps=(cycle - 1) * CYCLE_TIME_PS + cycle_step * DT_FS / 1000.0,
        event=event,
    )
    write(str(path), snapshot, format="extxyz", append=True)


class SimulationClock:
    """Keep ASE log time continuous across cleanup intervals and cycles."""

    def __init__(self, dyn, start_time):
        self.dyn = dyn
        self.start_time = start_time

    def get_time(self):
        return self.start_time + self.dyn.get_time()


def run_segment(atoms, steps, offset, cycle, output, md_handle, rng):
    # New dynamics object after deletion: masses/friction have the new size.
    friction = np.zeros((len(atoms), 1))
    friction[atoms.arrays["bath_atom"].astype(bool)] = FRICTION
    with Langevin(
        atoms, timestep=DT_FS * units.fs, temperature_K=TEMP_K,
        friction=friction, fixcm=False, rng=rng,
    ) as dyn:
        start_time = ((cycle - 1) * CYCLE_TIME_PS * 1000.0 + offset * DT_FS) * units.fs
        clock = SimulationClock(dyn, start_time)
        first_segment = cycle == 1 and offset == 0
        file_logger = MDLogger(clock, atoms, md_handle, header=first_segment)
        screen_logger = MDLogger(clock, atoms, "-", header=first_segment)
        dyn.attach(file_logger, interval=1)
        dyn.attach(screen_logger, interval=1)
        # Each segment logs its initial state, including the post-cleanup
        # state at the same time as the preceding segment's final state.
        last_written = -1

        def record():
            nonlocal last_written
            cycle_step = offset + dyn.nsteps
            # Explicit pre/post-cleanup frames capture the deletion event.
            if dyn.nsteps == 0:
                return
            save_frame(atoms, output / "implantation.extxyz", cycle, cycle_step, "md")
            last_written = dyn.nsteps

        dyn.attach(record, interval=WRITE_INTERVAL)
        dyn.run(steps)
        if last_written != dyn.nsteps:
            record()
        file_logger.close()
        screen_logger.close()
    return atoms


def run_cyclic_implantation():
    if DT_FS <= 0 or ENERGY_EV < 0 or CYCLES < 1:
        raise ValueError("Require DT_FS > 0, ENERGY_EV >= 0, CYCLES >= 1.")
    if not 0 < CLEANUP_TIME_PS < CYCLE_TIME_PS:
        raise ValueError("Cleanup must be strictly inside each cycle.")
    total_steps = steps_for(CYCLE_TIME_PS)
    cleanup_steps = steps_for(CLEANUP_TIME_PS)
    output = Path(OUTPUT_DIR)
    if output.exists():
        raise FileExistsError(f"Choose a new OUTPUT_DIR; {output} already exists.")
    rng = np.random.default_rng(RANDOM_SEED)
    system = read(SURFACE_XYZ)
    if not len(system):
        raise ValueError("Empty input surface.")

    if np.all(system.cell == 0):
        span = np.ptp(system.positions, axis=0)
        system.set_cell([span[0] + 0.1, span[1] + 0.1, max(span[2], 1.0)])
        system.center(axis=(0, 1))
    # This script assumes the slab normal is Cartesian z.
    cell = system.cell.copy()
    if not np.allclose(cell[:2, 2], 0) or not np.allclose(cell[2, :2], 0):
        raise ValueError("Orient the slab normal along z before using this script.")
    system.positions[:, 2] -= system.positions[:, 2].min()
    initial_top = system.positions[:, 2].max()
    removal_z = initial_top + SPUTTER_CLEARANCE_A
    cell[2, 2] = initial_top + 1000.0
    system.set_cell(cell)
    system.set_pbc([True, True, True])
    system.wrap(pbc=[True, True, False])

    fixed = system.positions[:, 2] < FIXED_CUTOFF_Z
    system.set_array("atom_id", np.arange(len(system), dtype=int))
    system.set_array("fixed_atom", fixed.astype(int))
    system.set_array("projectile", np.zeros(len(system), dtype=int))
    system.set_array("bath_atom", np.zeros(len(system), dtype=int))
    next_id = len(system)
    system.set_constraint(FixAtoms(mask=fixed))
    mobile = system[~fixed]
    mobile.set_constraint()
    if not len(mobile):
        raise ValueError("No mobile substrate atoms.")
    MaxwellBoltzmannDistribution(mobile, temperature_K=TEMP_K, force_temp=True, rng=rng)
    velocities = np.zeros((len(system), 3))
    velocities[~fixed] = mobile.get_velocities()
    system.set_velocities(velocities)

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    orbff, adapter = pretrained.orb_v3_direct_inf_omat(
        device=DEVICE, precision="float32-high",
    )
    calc = ORBCalculator(orbff, atoms_adapter=adapter, device=DEVICE)
    system.calc = calc
    system.info.update(charge=0, spin=1)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Each cycle: {total_steps} steps; cleanup every {cleanup_steps} steps.")
    print(f"Removal criterion: z > {removal_z:.3f} Å and vz > 0.")

    with (output / "sputtered.log").open("w", newline="") as sputter_log, \
         (output / "md.log").open("w", newline="") as md_log:
        sputter_writer = csv.writer(sputter_log)
        sputter_writer.writerow([
            "cycle", "cycle_time_ps", "total_time_ps", "removal_z_A",
            "n_removed", "removed_total_formula", "sputtered_substrate_formula",
            "removed_projectile_formula", "remaining_formula", "removed_atom_ids",
            "removed_atoms_json",
        ])
        for cycle in range(1, CYCLES + 1):
            old_v = system.get_velocities().copy()
            lengths = np.linalg.norm(cell[:2], axis=1)
            buffers = [3.0 / length if length > 6.0 else 0.1 for length in lengths]
            fx, fy = [rng.uniform(b, 1.0 - b) for b in buffers]
            ion_pos = fx * cell[0] + fy * cell[1]
            ion_pos[2] = system.positions[:, 2].max() + ION_HEIGHT_A
            ion = Atoms([IMPLANT_SPECIES], positions=[ion_pos])
            for key, value in [("atom_id", next_id), ("fixed_atom", 0),
                               ("projectile", 1), ("bath_atom", 0)]:
                ion.set_array(key, np.array([value], dtype=int))
            next_id += 1
            speed = np.sqrt(2.0 * ENERGY_EV / ion[0].mass)
            system += ion
            system.set_constraint(FixAtoms(mask=system.arrays["fixed_atom"].astype(bool)))
            system.set_velocities(np.vstack([old_v, [0.0, 0.0, -speed]]))
            # Determine bath membership once per cycle; retain it through cleanup.
            bath = ((system.positions[:, 2] >= FIXED_CUTOFF_Z)
                    & (system.positions[:, 2] < THERMO_CUTOFF_Z)
                    & ~system.arrays["fixed_atom"].astype(bool))
            bath[-1] = False
            system.set_array("bath_atom", bath.astype(int))
            print(f"Cycle {cycle}: {ENERGY_EV:g} eV {IMPLANT_SPECIES}, "
                  f"downward speed {speed * units.fs:.4f} Å/fs")
            save_frame(system, output / "implantation.extxyz", cycle, 0, "injection")
            offset = 0
            while offset < total_steps:
                segment_steps = min(cleanup_steps, total_steps - offset)
                system = run_segment(system, segment_steps, offset, cycle,
                                     output, md_log, rng)
                offset += segment_steps
                cleanup_time = offset * DT_FS / 1000.0
                system, removed = remove_sputtered(system, removal_z)
                is_projectile = removed.arrays["projectile"].astype(bool)
                details = [
                    {"atom_id": int(removed.arrays["atom_id"][i]),
                     "element": atom.symbol,
                     "projectile": bool(is_projectile[i]),
                     "position_A": atom.position.tolist(),
                     "velocity_A_per_fs": (removed.get_velocities()[i] * units.fs).tolist()}
                    for i, atom in enumerate(removed)
                ]
                # All cleanup events from all cycles append to this one log.
                sputter_writer.writerow([
                    cycle, cleanup_time, (cycle - 1) * CYCLE_TIME_PS + cleanup_time,
                    removal_z, len(removed), formula(removed),
                    formula(removed[~is_projectile]), formula(removed[is_projectile]),
                    formula(system), ";".join(map(str, removed.arrays["atom_id"])),
                    json.dumps(details),
                ])
                sputter_log.flush()
                save_frame(system, output / "implantation.extxyz", cycle,
                           offset, "after_cleanup")
                print(f"At {cleanup_time:g} ps: removed {len(removed)} atoms "
                      f"({formula(removed)}).")
    print(f"Complete. Outputs: {output.resolve()}")


if __name__ == "__main__":
    run_cyclic_implantation()