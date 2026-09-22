import os
import json
import shutil
import subprocess
import tempfile
import warnings

import numpy as np

from ase.io import read, write
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


warnings.filterwarnings("ignore")


# =================================================================
# 1. CONFIGURATION
# =================================================================

SURFACE_XYZ = "surface_tall.xyz"

IMPLANT_SPECIES = "Ar"
ENERGY_EV = 1000.0

CYCLES = 1

MD_STEPS_PER_CYCLE = 20000
DT_FS = 0.5

TEMP_K = 300.0

# Preserves your original friction value in inverse ASE time units.
FRICTION = 0.5


# -----------------------------------------------------------------
# FIXED / THERMOSTATTED REGIONS
# -----------------------------------------------------------------

FIXED_CUTOFF_Z = 3.0

THERMO_CUTOFF_Z = 100.0


# -----------------------------------------------------------------
# SPUTTER REMOVAL
# -----------------------------------------------------------------

# Check for sputtered material every 0.5 ps.
SPUTTER_INTERVAL_PS = 0.5

# Sputtered substrate atoms are defined as:
#
#     z > initial_surface_top + SPUTTER_BUFFER_Z
#
# Example:
# If the original surface top is 120 Å and this is 5 Å,
# atoms above 125 Å are considered sputtered.
#
# Set to 0.0 if you literally want the initial surface top
# to be the removal boundary.
SPUTTER_BUFFER_Z = 0.0

SPUTTER_LOG = "sputtered_species.log"


# -----------------------------------------------------------------
# IMPLANTED Ar FINAL COORDINATE
# -----------------------------------------------------------------

IMPLANT_LOG = "implant_final_coordinates.log"


# -----------------------------------------------------------------
# TRAJECTORY
# -----------------------------------------------------------------

OUTPUT_XYZ = "implantation.xyz"

# Write trajectory every 10 MD steps.
# At dt = 0.5 fs this is every 5 fs = 0.005 ps.
TRAJ_INTERVAL_STEPS = 10


# -----------------------------------------------------------------
# THERMODYNAMIC LOGGING
# -----------------------------------------------------------------

# Write:
#
# time
# kinetic energy
# potential energy
# total energy
# temperature
#
# every 10 MD steps.
THERMO_LOG_INTERVAL_STEPS = 10


# -----------------------------------------------------------------
# INTERNAL ASE ARRAYS
# -----------------------------------------------------------------

# Tracks which atom is the implanted ion.
#
# 0 = original/substrate atom
# n = ion introduced during implantation cycle n
IMPLANT_ID_ARRAY = "implant_cycle_id"

# Tracks the ORIGINAL fixed substrate atoms.
#
# This is safer than relying on atom indices because sputtered
# atom deletion changes ASE atom indices.
FIXED_FLAG_ARRAY = "fixed_substrate_atom"

# Tracks which substrate atoms belong to the thermostat bath
# during the current implantation cycle.
THERMO_FLAG_ARRAY = "thermostat_atom"


# =================================================================
# 2. OPEN BABEL
# =================================================================

def openbabel_smiles(atoms):
    """
    Convert a group of removed atoms to SMILES using Open Babel.

    Open Babel determines bonding from the XYZ geometry.

    Returns
    -------
    list[str]
        Individual disconnected SMILES fragments.

    Example
    -------
    ["[C]", "C=C", "O=C=O"]
    """

    if len(atoms) == 0:
        return []

    if shutil.which("obabel") is None:
        raise RuntimeError(
            "Open Babel executable 'obabel' was not found in PATH."
        )

    temp_xyz = None

    try:

        # ---------------------------------------------------------
        # Temporary XYZ containing only the removed atoms
        # ---------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".xyz",
            delete=False,
        ) as tmp:

            temp_xyz = tmp.name

        write(
            temp_xyz,
            atoms,
            format="xyz",
        )

        # ---------------------------------------------------------
        # XYZ -> SMILES
        # ---------------------------------------------------------

        result = subprocess.run(
            [
                "obabel",
                "-ixyz",
                temp_xyz,
                "-osmi",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout.strip()

        if not output:
            raise RuntimeError(
                "Open Babel returned an empty SMILES string."
            )

        # Open Babel normally gives:
        #
        # SMILES    filename
        #
        # We only want the SMILES field.
        first_line = output.splitlines()[0]

        full_smiles = first_line.split()[0]

        # Disconnected molecular fragments are separated by "."
        species = [
            fragment
            for fragment in full_smiles.split(".")
            if fragment
        ]

        return species

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            "Open Babel failed while converting sputtered "
            f"atoms to SMILES:\n{exc.stderr}"
        ) from exc

    finally:

        if (
            temp_xyz is not None
            and os.path.exists(temp_xyz)
        ):
            os.remove(temp_xyz)


# =================================================================
# 3. CONSTRAINT MANAGEMENT
# =================================================================

def apply_fixed_constraint(system):
    """
    Apply FixAtoms using the persistent fixed-atom flag.

    Atom indices change when sputtered atoms are deleted, so we
    should not permanently store fixed atom indices.
    """

    fixed_flags = system.arrays[
        FIXED_FLAG_ARRAY
    ]

    fixed_indices = np.flatnonzero(
        fixed_flags
    ).tolist()

    system.set_constraint(
        FixAtoms(
            indices=fixed_indices
        )
    )

    return fixed_indices


# =================================================================
# 4. SPUTTER REMOVAL
# =================================================================

def remove_sputtered_atoms(
    system,
    sputter_cutoff_z,
    implant_cycle,
    removal_cycle,
):
    """
    Remove sputtered SUBSTRATE atoms above the sputter cutoff.

    The implanted Ar is deliberately excluded from this removal.

    The removed atoms are converted to SMILES using Open Babel
    before deletion.
    """

    z_coords = system.positions[:, 2]

    implant_ids = system.arrays[
        IMPLANT_ID_ARRAY
    ]

    # -------------------------------------------------------------
    # Only substrate atoms are eligible for sputter removal.
    #
    # implant_ids == 0 -> substrate
    # implant_ids != 0 -> implanted ion
    # -------------------------------------------------------------

    sputter_mask = (
        (z_coords > sputter_cutoff_z)
        &
        (implant_ids == 0)
    )

    sputter_indices = np.flatnonzero(
        sputter_mask
    )

    # -------------------------------------------------------------
    # GET SMILES BEFORE DELETING
    # -------------------------------------------------------------

    if len(sputter_indices) > 0:

        sputtered_atoms = system[
            sputter_indices
        ].copy()

        species_smiles = openbabel_smiles(
            sputtered_atoms
        )

        # ASE atom deletion is safest without active constraints.
        system.set_constraint()

        del system[sputter_indices]

        # Rebuild fixed constraint because indices may have shifted.
        apply_fixed_constraint(
            system
        )

    else:

        species_smiles = []

    # -------------------------------------------------------------
    # LOG
    #
    # Only:
    #
    # implant cycle
    # removal cycle
    # removed SMILES list
    # -------------------------------------------------------------

    with open(
        SPUTTER_LOG,
        "a",
    ) as f:

        f.write(
            f"{implant_cycle}\t"
            f"{removal_cycle}\t"
            f"{json.dumps(species_smiles)}\n"
        )

    print(
        f"  Sputter removal {removal_cycle}: "
        f"{len(sputter_indices)} atoms removed "
        f"{species_smiles}"
    )

    return len(sputter_indices)


# =================================================================
# 5. REMOVE AND LOG IMPLANTED Ar
# =================================================================

def remove_and_log_implant(
    system,
    implant_cycle,
    implant_species,
):
    """
    At the end of an implantation cycle:

    1. Locate that cycle's implanted Ar.
    2. Record its final x, y, z coordinates.
    3. Remove it from the system.

    Raw/unwrapped MD coordinates are recorded.
    """

    implant_ids = system.arrays[
        IMPLANT_ID_ARRAY
    ]

    ion_indices = np.flatnonzero(
        implant_ids == implant_cycle
    )

    if len(ion_indices) != 1:

        raise RuntimeError(
            f"Expected exactly one implanted ion for cycle "
            f"{implant_cycle}, but found {len(ion_indices)}."
        )

    ion_index = int(
        ion_indices[0]
    )

    # -------------------------------------------------------------
    # RAW / UNWRAPPED FINAL POSITION
    # -------------------------------------------------------------

    x, y, z = system.positions[
        ion_index
    ].copy()

    # -------------------------------------------------------------
    # LOG COORDINATE
    # -------------------------------------------------------------

    with open(
        IMPLANT_LOG,
        "a",
    ) as f:

        f.write(
            f"{implant_cycle}\t"
            f"{implant_species}\t"
            f"{x:.10f}\t"
            f"{y:.10f}\t"
            f"{z:.10f}\n"
        )

    print(
        f"Final {implant_species} coordinate: "
        f"x={x:.6f}, "
        f"y={y:.6f}, "
        f"z={z:.6f} Å"
    )

    # -------------------------------------------------------------
    # REMOVE IMPLANTED Ar
    # -------------------------------------------------------------

    system.set_constraint()

    del system[ion_index]

    apply_fixed_constraint(
        system
    )


# =================================================================
# 6. THERMODYNAMIC LOG
# =================================================================

def initialize_cycle_log(cycle):
    """
    Create implant_cycle_N.log.
    """

    logfile = (
        f"implant_cycle_{cycle}.log"
    )

    with open(
        logfile,
        "w",
    ) as f:

        f.write(
            "time_ps\t"
            "kinetic_energy_eV\t"
            "potential_energy_eV\t"
            "total_energy_eV\t"
            "temperature_K\n"
        )

    return logfile


# =================================================================
# 7. MAIN SIMULATION
# =================================================================

def run_cyclic_implantation(
    surface_xyz,
    implant_species,
    energy_ev,
    cycles,
    steps,
    dt_fs,
):

    # =============================================================
    # VALIDATION
    # =============================================================

    if energy_ev < 0:

        raise ValueError(
            "Implantation energy must be nonnegative."
        )

    if SPUTTER_INTERVAL_PS <= 0:

        raise ValueError(
            "SPUTTER_INTERVAL_PS must be positive."
        )

    if shutil.which("obabel") is None:

        raise RuntimeError(
            "Open Babel executable 'obabel' was not found.\n"
            "Make sure Open Babel is installed and the "
            "'obabel' command is available in PATH."
        )

    # -------------------------------------------------------------
    # Convert sputter interval from ps -> number of MD steps
    # -------------------------------------------------------------

    removal_interval_float = (
        SPUTTER_INTERVAL_PS
        * 1000.0
        / dt_fs
    )

    removal_interval_steps = int(
        round(
            removal_interval_float
        )
    )

    if not np.isclose(
        removal_interval_float,
        removal_interval_steps,
    ):

        raise ValueError(
            "SPUTTER_INTERVAL_PS does not correspond "
            "to an integer number of MD steps."
        )

    print(
        f"Sputter removal interval: "
        f"{SPUTTER_INTERVAL_PS} ps = "
        f"{removal_interval_steps} MD steps."
    )

    # With:
    #
    # DT_FS = 0.5
    # SPUTTER_INTERVAL_PS = 0.5
    #
    # 0.5 ps = 500 fs
    #
    # 500 fs / 0.5 fs = 1000 MD steps


    # =============================================================
    # LOAD SURFACE
    # =============================================================

    print(
        f"Loading surface from {surface_xyz}..."
    )

    surface = read(
        surface_xyz
    )


    # =============================================================
    # CELL SETUP
    # =============================================================

    pos = surface.positions

    min_x = np.min(
        pos[:, 0]
    )

    max_x = np.max(
        pos[:, 0]
    )

    min_y = np.min(
        pos[:, 1]
    )

    max_y = np.max(
        pos[:, 1]
    )

    min_z = np.min(
        pos[:, 2]
    )

    max_z = np.max(
        pos[:, 2]
    )

    if np.all(
        surface.cell == 0
    ):

        xy_buffer = 0.1

        surface.set_cell(
            [
                max_x - min_x + xy_buffer,
                max_y - min_y + xy_buffer,
                max(
                    max_z - min_z,
                    1.0,
                ),
            ]
        )

        surface.center(
            axis=(0, 1)
        )


    # -------------------------------------------------------------
    # Shift substrate bottom to z = 0
    # -------------------------------------------------------------

    surface.positions[:, 2] -= np.min(
        surface.positions[:, 2]
    )


    # -------------------------------------------------------------
    # Large Z vacuum
    # -------------------------------------------------------------

    cell = surface.get_cell()

    initial_surface_top = np.max(
        surface.positions[:, 2]
    )

    cell[2, 2] = (
        initial_surface_top
        + 1000.0
    )

    surface.set_cell(
        cell
    )

    surface.set_pbc(
        [True, True, True]
    )


    # -------------------------------------------------------------
    # Wrap only X and Y.
    #
    # Preserve z = 0 reference.
    # -------------------------------------------------------------

    surface.wrap(
        pbc=[
            True,
            True,
            False,
        ]
    )


    # =============================================================
    # SPUTTER CUTOFF
    # =============================================================

    sputter_cutoff_z = (
        initial_surface_top
        + SPUTTER_BUFFER_Z
    )

    print(
        f"Initial surface top: "
        f"{initial_surface_top:.3f} Å"
    )

    print(
        f"Sputter cutoff: "
        f"z > {sputter_cutoff_z:.3f} Å"
    )


    # =============================================================
    # CREATE SYSTEM
    # =============================================================

    system = surface.copy()


    # -------------------------------------------------------------
    # Persistent fixed-substrate flag
    # -------------------------------------------------------------

    fixed_flags = (
        system.positions[:, 2]
        < FIXED_CUTOFF_Z
    )

    system.set_array(
        FIXED_FLAG_ARRAY,
        fixed_flags.astype(bool),
    )


    # -------------------------------------------------------------
    # Every original atom is substrate -> implant ID = 0
    # -------------------------------------------------------------

    system.set_array(
        IMPLANT_ID_ARRAY,
        np.zeros(
            len(system),
            dtype=np.int64,
        ),
    )


    # -------------------------------------------------------------
    # Thermostat flag will be populated at the beginning of
    # every implantation cycle.
    # -------------------------------------------------------------

    system.set_array(
        THERMO_FLAG_ARRAY,
        np.zeros(
            len(system),
            dtype=bool,
        ),
    )


    # -------------------------------------------------------------
    # Apply fixed constraint
    # -------------------------------------------------------------

    fixed_indices = apply_fixed_constraint(
        system
    )

    print(
        f"Anchored bottom {FIXED_CUTOFF_Z} Å "
        f"({len(fixed_indices)} fixed atoms)."
    )

    print(
        f"Thermostat layer: "
        f"{FIXED_CUTOFF_Z} Å <= z "
        f"< {THERMO_CUTOFF_Z} Å "
        f"at {TEMP_K} K."
    )


    # =============================================================
    # INITIALIZE SUBSTRATE VELOCITIES
    # =============================================================

    mobile_mask = ~system.arrays[
        FIXED_FLAG_ARRAY
    ]

    if not np.any(
        mobile_mask
    ):

        raise ValueError(
            "No mobile substrate atoms remain "
            "above the fixed layer."
        )

    mobile_atoms = system[
        mobile_mask
    ]

    mobile_atoms.set_constraint()

    MaxwellBoltzmannDistribution(
        mobile_atoms,
        temperature_K=TEMP_K,
        force_temp=True,
    )


    # -------------------------------------------------------------
    # Fixed atoms stay at zero velocity
    # -------------------------------------------------------------

    velocities = np.zeros(
        (
            len(system),
            3,
        )
    )

    velocities[
        mobile_mask
    ] = mobile_atoms.get_velocities()

    system.set_velocities(
        velocities
    )

    print(
        f"Initialized {len(mobile_atoms)} "
        f"mobile substrate atoms at "
        f"{mobile_atoms.get_temperature():.2f} K."
    )


    # =============================================================
    # ML FORCE FIELD
    # =============================================================

    device = "cuda"

    print(
        "Loading ORB-v3 direct OMat force field..."
    )

    orbff = (
        pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
        )
    )

    calc = ORBCalculator(
        orbff,
        device=device,
    )


    # =============================================================
    # OUTPUT FILES
    # =============================================================

    for filename in (
        OUTPUT_XYZ,
        SPUTTER_LOG,
        IMPLANT_LOG,
    ):

        if os.path.exists(
            filename
        ):
            os.remove(
                filename
            )


    # -------------------------------------------------------------
    # SPUTTER LOG HEADER
    # -------------------------------------------------------------

    with open(
        SPUTTER_LOG,
        "w",
    ) as f:

        f.write(
            "implant_cycle\t"
            "removal_cycle\t"
            "removed_smiles\n"
        )


    # -------------------------------------------------------------
    # IMPLANT POSITION LOG HEADER
    # -------------------------------------------------------------

    with open(
        IMPLANT_LOG,
        "w",
    ) as f:

        f.write(
            "implant_cycle\t"
            "species\t"
            "x_A\t"
            "y_A\t"
            "z_A\n"
        )


    # =============================================================
    # IMPLANTATION CYCLES
    # =============================================================

    for cycle in range(
        1,
        cycles + 1,
    ):

        print(
            f"\n========================================"
        )

        print(
            f"Implantation Cycle {cycle}/{cycles}"
        )

        print(
            f"========================================"
        )


        # =========================================================
        # THERMOSTAT MEMBERSHIP FOR THIS CYCLE
        # =========================================================

        z_coords = system.positions[
            :,
            2
        ]

        fixed_flags = system.arrays[
            FIXED_FLAG_ARRAY
        ]

        implant_ids = system.arrays[
            IMPLANT_ID_ARRAY
        ]

        thermo_flags = (
            (z_coords >= FIXED_CUTOFF_Z)
            &
            (z_coords < THERMO_CUTOFF_Z)
            &
            (~fixed_flags)
            &
            (implant_ids == 0)
        )

        system.set_array(
            THERMO_FLAG_ARRAY,
            thermo_flags.astype(bool),
        )

        print(
            f"Thermostatted substrate atoms: "
            f"{np.count_nonzero(thermo_flags)}"
        )


        # =========================================================
        # RANDOM IMPLANT POSITION
        # =========================================================

        current_max_z = np.max(
            system.positions[:, 2]
        )

        cell = system.get_cell()

        edge_buffer = 3.0

        len_x = np.linalg.norm(
            cell[0]
        )

        len_y = np.linalg.norm(
            cell[1]
        )

        frac_buf_x = (
            edge_buffer / len_x
            if len_x > 2 * edge_buffer
            else 0.1
        )

        frac_buf_y = (
            edge_buffer / len_y
            if len_y > 2 * edge_buffer
            else 0.1
        )

        fx = np.random.uniform(
            frac_buf_x,
            1.0 - frac_buf_x,
        )

        fy = np.random.uniform(
            frac_buf_y,
            1.0 - frac_buf_y,
        )

        ion_cart = (
            fx * cell[0]
            + fy * cell[1]
        )

        # Preserve your current placement:
        # 0.5 Å above the highest existing atom.
        ion_cart[2] = (
            current_max_z
            + 0.5
        )


        # =========================================================
        # CREATE IMPLANT ATOM
        # =========================================================

        implant_atom = Atoms(
            symbols=[
                implant_species
            ],
            positions=[
                ion_cart
            ],
        )


        # ---------------------------------------------------------
        # Mark this atom with the current implantation cycle.
        # ---------------------------------------------------------

        implant_atom.set_array(
            IMPLANT_ID_ARRAY,
            np.array(
                [cycle],
                dtype=np.int64,
            ),
        )


        # ---------------------------------------------------------
        # Implant is never a fixed substrate atom.
        # ---------------------------------------------------------

        implant_atom.set_array(
            FIXED_FLAG_ARRAY,
            np.array(
                [False],
                dtype=bool,
            ),
        )


        # ---------------------------------------------------------
        # Implant is never thermostatted.
        # ---------------------------------------------------------

        implant_atom.set_array(
            THERMO_FLAG_ARRAY,
            np.array(
                [False],
                dtype=bool,
            ),
        )


        # =========================================================
        # ION VELOCITY FROM IMPLANTATION ENERGY
        # =========================================================

        mass = implant_atom[
            0
        ].mass

        # ASE internal velocity units.
        #
        # E[eV] = 0.5 * mass[amu] * v_internal^2
        v_mag = np.sqrt(
            2.0
            * energy_ev
            / mass
        )

        ion_velocities = np.zeros(
            (
                1,
                3,
            )
        )

        ion_velocities[
            0,
            2
        ] = -v_mag

        ion_ke = (
            0.5
            * mass
            * np.sum(
                ion_velocities ** 2
            )
        )

        # Convert ASE internal velocity to physical Å/fs
        # for printing only.
        v_ang_per_fs = (
            v_mag
            * units.fs
        )

        print(
            f"Incoming {implant_species}:"
        )

        print(
            f"  kinetic energy = "
            f"{ion_ke:.2f} eV"
        )

        print(
            f"  velocity in -Z = "
            f"{v_ang_per_fs:.4f} Å/fs"
        )


        # =========================================================
        # ADD IMPLANT TO SYSTEM
        # =========================================================

        old_velocities = (
            system
            .get_velocities()
            .copy()
        )

        system += implant_atom

        apply_fixed_constraint(
            system
        )

        system.set_velocities(
            np.vstack(
                (
                    old_velocities,
                    ion_velocities,
                )
            )
        )

        system.calc = calc

        system.info[
            "charge"
        ] = 0

        system.info[
            "spin"
        ] = 1


        # =========================================================
        # CREATE THIS CYCLE'S THERMODYNAMIC LOG
        # =========================================================

        cycle_logfile = initialize_cycle_log(
            cycle
        )


        # =========================================================
        # RUN MD IN 0.5 ps CHUNKS
        # =========================================================

        completed_steps = 0

        removal_cycle = 0

        # Prevent duplicate frames/log entries when a new
        # Langevin object begins at each 0.5 ps boundary.
        last_written_step = None

        last_thermo_step = None


        print(
            f"Running {steps} MD steps "
            f"with dt = {dt_fs} fs."
        )

        print(
            f"Total cycle time = "
            f"{steps * dt_fs / 1000.0:.3f} ps."
        )

        print(
            f"Sputter cleanup every "
            f"{SPUTTER_INTERVAL_PS} ps."
        )


        # =========================================================
        # MD LOOP
        # =========================================================

        while completed_steps < steps:

            chunk_steps = min(
                removal_interval_steps,
                steps - completed_steps,
            )


            # =====================================================
            # REBUILD CONSTRAINT INDICES
            # =====================================================

            fixed_indices = apply_fixed_constraint(
                system
            )


            # =====================================================
            # THERMOSTAT FRICTION ARRAY
            # =====================================================

            thermo_flags = system.arrays[
                THERMO_FLAG_ARRAY
            ]

            friction_array = np.zeros(
                (
                    len(system),
                    1,
                )
            )

            friction_array[
                thermo_flags
            ] = FRICTION


            # =====================================================
            # CREATE LANGEVIN OBJECT
            # =====================================================
            #
            # We recreate Langevin after each 0.5 ps section
            # because the number of atoms may change after
            # sputter removal.
            # =====================================================

            dyn = Langevin(
                system,
                timestep=(
                    dt_fs
                    * units.fs
                ),
                temperature_K=TEMP_K,
                friction=friction_array,
                fixcm=False,
                logfile=None,
            )


            # -----------------------------------------------------
            # This is the global cycle step at which this chunk
            # begins.
            # -----------------------------------------------------

            chunk_start_step = completed_steps


            # =====================================================
            # TRAJECTORY WRITER
            # =====================================================

            def write_native_extxyz():

                nonlocal last_written_step

                global_step = (
                    chunk_start_step
                    + dyn.nsteps
                )

                # ASE executes observers at nsteps = 0 when a new
                # dynamics object begins. Avoid duplicate boundary
                # frames.
                if (
                    global_step
                    == last_written_step
                ):
                    return

                atoms_copy = system.copy()

                # Preserve your original wrapped trajectory output.
                atoms_copy.wrap()

                write(
                    OUTPUT_XYZ,
                    atoms_copy,
                    format="extxyz",
                    append=True,
                )

                last_written_step = (
                    global_step
                )


            # =====================================================
            # THERMODYNAMIC LOGGER
            # =====================================================

            def write_thermo_log():

                nonlocal last_thermo_step

                global_step = (
                    chunk_start_step
                    + dyn.nsteps
                )

                # Avoid duplicate entries at 0.5 ps boundaries.
                if (
                    global_step
                    == last_thermo_step
                ):
                    return

                # -------------------------------------------------
                # Simulation time
                # -------------------------------------------------

                time_ps = (
                    global_step
                    * dt_fs
                    / 1000.0
                )

                # -------------------------------------------------
                # Energies
                # -------------------------------------------------

                kinetic_energy = (
                    system.get_kinetic_energy()
                )

                potential_energy = (
                    system.get_potential_energy()
                )

                total_energy = (
                    kinetic_energy
                    + potential_energy
                )

                # -------------------------------------------------
                # Temperature
                # -------------------------------------------------

                temperature = (
                    system.get_temperature()
                )

                # -------------------------------------------------
                # Write log
                # -------------------------------------------------

                with open(
                    cycle_logfile,
                    "a",
                ) as f:

                    f.write(
                        f"{time_ps:.6f}\t"
                        f"{kinetic_energy:.10f}\t"
                        f"{potential_energy:.10f}\t"
                        f"{total_energy:.10f}\t"
                        f"{temperature:.6f}\n"
                    )

                last_thermo_step = (
                    global_step
                )


            # =====================================================
            # ATTACH OUTPUT FUNCTIONS
            # =====================================================

            dyn.attach(
                write_native_extxyz,
                interval=TRAJ_INTERVAL_STEPS,
            )

            dyn.attach(
                write_thermo_log,
                interval=THERMO_LOG_INTERVAL_STEPS,
            )


            # =====================================================
            # RUN CURRENT SECTION
            # =====================================================

            dyn.run(
                chunk_steps
            )

            completed_steps += (
                chunk_steps
            )


            # =====================================================
            # SPUTTER REMOVAL
            # =====================================================
            #
            # Only perform a removal cycle after a COMPLETE
            # SPUTTER_INTERVAL_PS interval.
            #
            # With your present settings every chunk is 1000 steps,
            # so this happens at:
            #
            # 0.5 ps
            # 1.0 ps
            # 1.5 ps
            # 2.0 ps
            # ...
            # =====================================================

            if (
                chunk_steps
                == removal_interval_steps
            ):

                removal_cycle += 1

                remove_sputtered_atoms(
                    system=system,
                    sputter_cutoff_z=(
                        sputter_cutoff_z
                    ),
                    implant_cycle=cycle,
                    removal_cycle=(
                        removal_cycle
                    ),
                )

                # Reattach calculator after changing atom count.
                system.calc = calc

                system.info[
                    "charge"
                ] = 0

                system.info[
                    "spin"
                ] = 1


        # =========================================================
        # WRITE FINAL TRAJECTORY FRAME IF NECESSARY
        # =========================================================

        if (
            completed_steps
            != last_written_step
        ):

            atoms_copy = system.copy()

            atoms_copy.wrap()

            write(
                OUTPUT_XYZ,
                atoms_copy,
                format="extxyz",
                append=True,
            )

            last_written_step = (
                completed_steps
            )


        # =========================================================
        # WRITE FINAL THERMO STATE IF NECESSARY
        # =========================================================

        if (
            completed_steps
            != last_thermo_step
        ):

            time_ps = (
                completed_steps
                * dt_fs
                / 1000.0
            )

            kinetic_energy = (
                system.get_kinetic_energy()
            )

            potential_energy = (
                system.get_potential_energy()
            )

            total_energy = (
                kinetic_energy
                + potential_energy
            )

            temperature = (
                system.get_temperature()
            )

            with open(
                cycle_logfile,
                "a",
            ) as f:

                f.write(
                    f"{time_ps:.6f}\t"
                    f"{kinetic_energy:.10f}\t"
                    f"{potential_energy:.10f}\t"
                    f"{total_energy:.10f}\t"
                    f"{temperature:.6f}\n"
                )


        # =========================================================
        # END OF IMPLANTATION CYCLE:
        #
        # 1. RECORD FINAL Ar XYZ
        # 2. REMOVE Ar
        # =========================================================

        remove_and_log_implant(
            system=system,
            implant_cycle=cycle,
            implant_species=(
                implant_species
            ),
        )

        system.calc = calc

        system.info[
            "charge"
        ] = 0

        system.info[
            "spin"
        ] = 1


        print(
            f"Cycle {cycle} complete!"
        )

        print(
            f"Thermodynamic log: "
            f"'{cycle_logfile}'"
        )


    # =============================================================
    # COMPLETE
    # =============================================================

    print(
        f"\nAll {cycles} cycles complete!"
    )

    print(
        f"Trajectory: "
        f"'{OUTPUT_XYZ}'"
    )

    print(
        f"Sputtered species log: "
        f"'{SPUTTER_LOG}'"
    )

    print(
        f"Implant coordinate log: "
        f"'{IMPLANT_LOG}'"
    )


# =================================================================
# 8. RUN
# =================================================================

if __name__ == "__main__":

    run_cyclic_implantation(
        surface_xyz=SURFACE_XYZ,
        implant_species=IMPLANT_SPECIES,
        energy_ev=ENERGY_EV,
        cycles=CYCLES,
        steps=MD_STEPS_PER_CYCLE,
        dt_fs=DT_FS,
    )