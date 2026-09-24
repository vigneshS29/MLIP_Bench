import os
import numpy as np

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from mace.calculators import mace_off


# =====================================================================
# 1. USER SETTINGS
# =====================================================================

INPUT_XYZ = "temp.xyz"

TRAJ_XYZ = "traj.xyz"
LOG_FILE = "md.log"


# ---------------------------------------------------------------------
# MD SETTINGS
# ---------------------------------------------------------------------

N_STEPS = 2000

DT_FS = 0.5

TEMPERATURE_K = 300.0

# Exact standard atmosphere
PRESSURE_PA = 101325.0


# ---------------------------------------------------------------------
# OUTPUT SETTINGS
# ---------------------------------------------------------------------

# Write wrapped XYZ every 1000 MD steps
TRAJ_INTERVAL = 1000

# Write thermo information every MD step
LOG_INTERVAL = 10


# ---------------------------------------------------------------------
# CELL
# ---------------------------------------------------------------------

# Buffer added on EACH side of min/max atomic coordinates
CELL_BUFFER_A = 0.1


# ---------------------------------------------------------------------
# THERMOSTAT / BAROSTAT
# ---------------------------------------------------------------------

# ASE recommends approximately:
#
# tdamp ~ 100 timesteps
# pdamp ~ 1000 timesteps
#
# For dt = 0.5 fs:
#
# 100 timesteps  = 50 fs
# 1000 timesteps = 500 fs
#
TDAMP_FS = 50.0
PDAMP_FS = 500.0


# ---------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------

DEVICE = "cuda"

DEFAULT_DTYPE = "float32"

# Explicit MACE-OFF24 medium model.
#
# IMPORTANT:
# mace_off(model="medium")
# currently corresponds to MACE-OFF23 medium.
#
# Therefore explicitly provide the MACE-OFF24 model.
#
MACE_OFF24_MODEL = (
    "https://raw.githubusercontent.com/"
    "ACEsuit/mace-off/main/"
    "mace_off23/MACE-OFF23_small.model"
)


# ---------------------------------------------------------------------
# RANDOM VELOCITY SEED
# ---------------------------------------------------------------------

RANDOM_SEED = 12345


# =====================================================================
# 2. READ STRUCTURE
# =====================================================================

print("\nReading structure...")

atoms = read(INPUT_XYZ)

print(f"Number of atoms: {len(atoms)}")


# =====================================================================
# 3. INFER CELL FROM ATOMIC COORDINATES
# =====================================================================

positions = atoms.get_positions()

xyz_min = positions.min(axis=0)
xyz_max = positions.max(axis=0)

span = xyz_max - xyz_min


print("\nOriginal coordinate limits:")

print(
    f"X: {xyz_min[0]:12.6f} -> "
    f"{xyz_max[0]:12.6f} A"
)

print(
    f"Y: {xyz_min[1]:12.6f} -> "
    f"{xyz_max[1]:12.6f} A"
)

print(
    f"Z: {xyz_min[2]:12.6f} -> "
    f"{xyz_max[2]:12.6f} A"
)


# ---------------------------------------------------------------------
# Add 0.1 A on BOTH sides
# ---------------------------------------------------------------------

cell_lengths = span + 2.0 * CELL_BUFFER_A

Lx = cell_lengths[0]
Ly = cell_lengths[1]
Lz = cell_lengths[2]


# ---------------------------------------------------------------------
# Shift atoms
#
# After shifting:
#
# minimum coordinate = 0.1 A
# maximum coordinate = L - 0.1 A
#
# ---------------------------------------------------------------------

shift = np.array(
    [
        CELL_BUFFER_A - xyz_min[0],
        CELL_BUFFER_A - xyz_min[1],
        CELL_BUFFER_A - xyz_min[2],
    ]
)

atoms.translate(shift)


# ---------------------------------------------------------------------
# Create orthorhombic simulation cell
# ---------------------------------------------------------------------

atoms.set_cell(
    [
        [Lx, 0.0, 0.0],
        [0.0, Ly, 0.0],
        [0.0, 0.0, Lz],
    ],
    scale_atoms=False,
)

atoms.set_pbc([True, True, True])


# ---------------------------------------------------------------------
# Check resulting coordinates
# ---------------------------------------------------------------------

positions = atoms.get_positions()

new_min = positions.min(axis=0)
new_max = positions.max(axis=0)


print("\nInferred simulation cell:")

print(f"Lx = {Lx:.6f} A")
print(f"Ly = {Ly:.6f} A")
print(f"Lz = {Lz:.6f} A")

print(
    f"Initial volume = "
    f"{atoms.get_volume():.6f} A^3"
)


print("\nCoordinates after shifting:")

print(
    f"X: {new_min[0]:12.6f} -> "
    f"{new_max[0]:12.6f} A"
)

print(
    f"Y: {new_min[1]:12.6f} -> "
    f"{new_max[1]:12.6f} A"
)

print(
    f"Z: {new_min[2]:12.6f} -> "
    f"{new_max[2]:12.6f} A"
)


# =====================================================================
# 4. CHECK MACE-OFF ELEMENT SUPPORT
# =====================================================================

allowed_elements = {
    "H",
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
}

present_elements = set(atoms.get_chemical_symbols())

unsupported_elements = present_elements - allowed_elements


print("\nElements present:")

print(sorted(present_elements))


if unsupported_elements:

    raise RuntimeError(
        "\nMACE-OFF24 does not support all elements "
        "in this structure.\n\n"
        f"Unsupported elements: "
        f"{sorted(unsupported_elements)}\n"
    )


# =====================================================================
# 5. LOAD MACE-OFF24
# =====================================================================

print("\nLoading MACE-OFF24(M)...")

calculator = mace_off(
    model=MACE_OFF24_MODEL,
    device=DEVICE,
    default_dtype=DEFAULT_DTYPE,
)

atoms.calc = calculator

print("MACE-OFF24(M) loaded successfully.")


# =====================================================================
# 6. INITIALIZE VELOCITIES
# =====================================================================

print(
    f"\nInitializing velocities at "
    f"{TEMPERATURE_K:.1f} K..."
)

rng = np.random.default_rng(RANDOM_SEED)


MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=TEMPERATURE_K,
    rng=rng,
    force_temp=True,
)


# Remove center-of-mass linear momentum
Stationary(atoms)


# Remove overall angular momentum
ZeroRotation(atoms)


print(
    f"Initial temperature = "
    f"{atoms.get_temperature():.6f} K"
)


# =====================================================================
# 7. PRESSURE CONVERSION
# =====================================================================

# ASE NPT requires pressure in eV / A^3.
#
# units.Pascal converts Pa -> ASE pressure units.
#
pressure_au = PRESSURE_PA * units.Pascal


print("\nTarget conditions:")

print(
    f"Temperature = "
    f"{TEMPERATURE_K:.2f} K"
)

print(
    f"Pressure    = "
    f"{PRESSURE_PA:.2f} Pa"
)

print(
    f"Pressure    = "
    f"{pressure_au:.10e} eV/A^3"
)

print(
    "Pressure    = 1.000000 atm"
)


# =====================================================================
# 8. SET UP ISOTROPIC NPT
# =====================================================================

print("\nCreating isotropic MTK NPT dynamics...")


dyn = IsotropicMTKNPT(

    atoms,

    timestep=DT_FS * units.fs,

    temperature_K=TEMPERATURE_K,

    pressure_au=pressure_au,

    tdamp=TDAMP_FS * units.fs,

    pdamp=PDAMP_FS * units.fs,

    tchain=3,

    pchain=3,

    tloop=1,

    ploop=1,
)


# ASE's MTK NPT interface uses pressure in eV/A^3 and defines tdamp and
# pdamp as characteristic thermostat/barostat times. Typical suggested
# values are ~100 and ~1000 timesteps, respectively.
# https://docs.ase-lib.org/_modules/ase/md/nose_hoover_chain.html


# =====================================================================
# 9. REMOVE PREVIOUS OUTPUT FILES
# =====================================================================

for filename in [TRAJ_XYZ, LOG_FILE]:

    if os.path.exists(filename):

        os.remove(filename)


# =====================================================================
# 10. OPEN MD LOG
# =====================================================================

log_handle = open(
    LOG_FILE,
    "w",
    buffering=1,
)


log_handle.write(

    "# step "
    "time_ps "
    "temperature_K "
    "pressure_atm "
    "density_g_cm3\n"
)


# =====================================================================
# 11. UNIT CONVERSION FOR DENSITY
# =====================================================================

# 1 amu / A^3 = 1.66053906660 g/cm^3

AMU_A3_TO_G_CM3 = 1.66053906660


# =====================================================================
# 12. FUNCTION: INSTANTANEOUS PRESSURE
# =====================================================================

def get_pressure_atm():

    """
    Return instantaneous pressure in atmospheres.

    ASE stress sign convention:

        pressure = -trace(stress) / 3

    include_ideal_gas=True adds the kinetic contribution.
    """

    stress = atoms.get_stress(
        voigt=False,
        include_ideal_gas=True,
    )

    pressure_ev_a3 = (
        -np.trace(stress) / 3.0
    )

    pressure_atm = (
        pressure_ev_a3 / pressure_au
    )

    return pressure_atm


# =====================================================================
# 13. FUNCTION: DENSITY
# =====================================================================

def get_density():

    """
    Return instantaneous density in g/cm^3.
    """

    total_mass_amu = (
        atoms.get_masses().sum()
    )

    volume_a3 = atoms.get_volume()

    density = (
        total_mass_amu
        * AMU_A3_TO_G_CM3
        / volume_a3
    )

    return density


# =====================================================================
# 14. FUNCTION: WRITE MD LOG
# =====================================================================

def write_md_log():

    step = dyn.nsteps

    time_ps = (
        dyn.get_time()
        / (1000.0 * units.fs)
    )

    temperature = (
        atoms.get_temperature()
    )

    pressure_atm = (
        get_pressure_atm()
    )

    density = (
        get_density()
    )


    log_handle.write(

        f"{step:10d} "

        f"{time_ps:15.8f} "

        f"{temperature:15.6f} "

        f"{pressure_atm:18.8f} "

        f"{density:18.8f}\n"
    )


# =====================================================================
# 15. FUNCTION: WRITE WRAPPED XYZ
# =====================================================================

def write_trajectory():

    """
    Write a wrapped snapshot to traj.xyz.

    A copy of the atoms object is wrapped so that wrapping
    does NOT modify the actual MD coordinates.
    """

    frame = atoms.copy()


    # -------------------------------------------------------------
    # Wrap XYZ coordinates into current NPT cell
    # -------------------------------------------------------------

    frame.wrap()


    # -------------------------------------------------------------
    # Current thermodynamic values
    # -------------------------------------------------------------

    step = dyn.nsteps

    time_ps = (
        dyn.get_time()
        / (1000.0 * units.fs)
    )

    temperature = (
        atoms.get_temperature()
    )

    pressure_atm = (
        get_pressure_atm()
    )

    density = (
        get_density()
    )


    # -------------------------------------------------------------
    # Store information in EXTXYZ comment line
    # -------------------------------------------------------------

    frame.info["step"] = step

    frame.info["time_ps"] = time_ps

    frame.info["temperature_K"] = temperature

    frame.info["pressure_atm"] = pressure_atm

    frame.info["density_g_cm3"] = density


    # -------------------------------------------------------------
    # Append frame
    #
    # EXTXYZ automatically stores:
    #
    # Lattice="..."
    # pbc="T T T"
    #
    # in the XYZ comment line.
    # -------------------------------------------------------------

    write(
        TRAJ_XYZ,
        frame,
        format="extxyz",
        append=True,
    )


# =====================================================================
# 16. ATTACH OUTPUT ROUTINES
# =====================================================================

# Thermodynamic information every MD step
dyn.attach(
    write_md_log,
    interval=LOG_INTERVAL,
)


# Wrapped XYZ every 1000 MD steps
dyn.attach(
    write_trajectory,
    interval=TRAJ_INTERVAL,
)


# =====================================================================
# 17. PRINT RUN INFORMATION
# =====================================================================

total_time_ps = (
    N_STEPS
    * DT_FS
    / 1000.0
)


print("\n====================================================")
print("                 NPT RUN SETTINGS")
print("====================================================")

print(
    f"Input structure       : "
    f"{INPUT_XYZ}"
)

print(
    f"Number of atoms       : "
    f"{len(atoms)}"
)

print(
    f"MACE model            : "
    f"MACE-OFF24(M)"
)

print(
    f"Device                : "
    f"{DEVICE}"
)

print(
    f"Temperature           : "
    f"{TEMPERATURE_K:.2f} K"
)

print(
    "Pressure              : "
    "1.0 atm"
)

print(
    f"Timestep              : "
    f"{DT_FS:.3f} fs"
)

print(
    f"Number of steps       : "
    f"{N_STEPS}"
)

print(
    f"Simulation time       : "
    f"{total_time_ps:.6f} ps"
)

print(
    f"Thermostat damping    : "
    f"{TDAMP_FS:.3f} fs"
)

print(
    f"Barostat damping      : "
    f"{PDAMP_FS:.3f} fs"
)

print(
    f"Initial cell          : "
    f"{Lx:.6f} x "
    f"{Ly:.6f} x "
    f"{Lz:.6f} A"
)

print(
    f"Initial density       : "
    f"{get_density():.6f} g/cm^3"
)

print(
    f"Trajectory interval   : "
    f"{TRAJ_INTERVAL} steps"
)

print(
    f"Log interval          : "
    f"{LOG_INTERVAL} step"
)

print("====================================================")


# =====================================================================
# 18. RUN NPT
# =====================================================================

print("\nStarting NPT simulation...\n")


try:

    dyn.run(N_STEPS)


finally:

    log_handle.close()


# =====================================================================
# 19. FINISHED
# =====================================================================

print("\n====================================================")
print("                 SIMULATION COMPLETE")
print("====================================================")

print(
    f"Trajectory written to : "
    f"{TRAJ_XYZ}"
)

print(
    f"MD log written to     : "
    f"{LOG_FILE}"
)

print(
    f"Final cell volume     : "
    f"{atoms.get_volume():.6f} A^3"
)

print(
    f"Final density         : "
    f"{get_density():.6f} g/cm^3"
)

print(
    f"Final temperature     : "
    f"{atoms.get_temperature():.6f} K"
)

print(
    f"Final pressure        : "
    f"{get_pressure_atm():.6f} atm"
)

print("====================================================")