# MOF Poromechanical Models

This repository contains molecular-dynamics inputs and pressure-driven poromechanical models for HKUST-1 and MIL-53(Cr). Both dry and water-containing systems are included. The Python models generate loading-unloading stress-strain curves, while the LAMMPS inputs provide the corresponding atomistic simulation setups.

## Repository structure

```text
.
|-- MD simulation/
|   |-- with water/
|   |   |-- HKUST-1/
|   |   `-- MIL-53/
|   `-- without water/
|       |-- HKUST-1/
|       `-- MIL-53/
`-- poromechanical-model/
    |-- HKUST-1/
    |   |-- load-unload/
    |   |   |-- with water/
    |   |   `-- without water/
    |   `-- pa/
    `-- MIL-53/
        |-- load-unload/
        |   |-- with water/
        |   `-- without water/
        `-- pa/
```

- `MD simulation/` contains LAMMPS input files, initial structures, and the HKUST-1 force-field file.
- `poromechanical-model/*/load-unload/` contains the constitutive models and editable parameter files.
- `poromechanical-model/*/pa/` evaluates and plots adsorption stress as a function of strain.

## Python requirements

Python 3.10 or later is recommended. Install the numerical dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```

Each material and hydration state is self-contained. Run a model from the directory containing its model and configuration files.

## Poromechanical models

### HKUST-1 with water

```bash
cd "poromechanical-model/HKUST-1/load-unload/with water"
python hkust1_model.py
```

The default water loading is set by `N_VALUE` in `hkust1_config.py`. To override it from the command line:

```bash
python hkust1_model.py --n 32
python hkust1_model.py --n 32 --show
python hkust1_model.py --n 32 --inspect
```

This model uses a one-way survival probability, logarithmic hardening, and irreversible unloading.

### Dry HKUST-1

```bash
cd "poromechanical-model/HKUST-1/load-unload/without water"
python hkust1_dry_model.py
```

The dry model uses the accepted onset-continuation construction after the open-pore stability limit. Its closed-pore modulus and transformation strain are direct inputs in `hkust1_dry_config.py`.

### MIL-53 with water

```bash
cd "poromechanical-model/MIL-53/load-unload/with water"
python mil53_model.py
```

The default water loading is set by `N_VALUE` in `mil53_config.py`. It can also be supplied at run time:

```bash
python mil53_model.py --n 5
python mil53_model.py --n 5 --show
python mil53_model.py --n 5 --inspect
```

This model uses a two-way survival probability, quadratic hardening, and reversible phase transformation.

### Dry MIL-53

```bash
cd "poromechanical-model/MIL-53/load-unload/without water"
python mil53_dry_model.py
```

The dry closed-pore modulus and transformation strain are direct inputs in `mil53_dry_config.py`. Reverse transformation is retained during unloading.


### Model output

Each loading-unloading run writes one figure:

```text
output/N=<water-loading>/stress-strain.png
```

The scripts also perform internal checks on calibration residuals, constitutive stress residuals, phase-fraction bounds, and transformation direction before reporting a successful run.

## Adsorption-stress curves

Run the direct adsorption-stress scripts from their respective directories:

```bash
cd "poromechanical-model/HKUST-1/pa"
python plot_pa_eps_hkust1_direct.py
```

```bash
cd "poromechanical-model/MIL-53/pa"
python plot_pa_eps_mil53_direct.py
```

Each script writes a PNG figure and a tab-separated text table beside the script. The arrays in `STRAIN_RANGES` control the evaluated water loadings and maximum strains.

## LAMMPS simulations

Run each input from its own directory so that relative structure and force-field paths resolve correctly. For example:

```bash
cd "MD simulation/with water/HKUST-1"
lmp -in in.water
```

The exact executable name may be `lmp`, `lmp_mpi`, or installation-specific. The HKUST-1 inputs require a LAMMPS build with ReaxFF and charge-equilibration support. The hydrated MIL-53 input requires the TIP4P pair style used in `in.water`.

Hydrated structures supplied in this repository correspond to the following water loadings:

- HKUST-1: `N = 8, 16, 24, 32`
- MIL-53: `N = 1, 2, 3, 4, 5`

## Conventions

- Compression, pressure, and compressive strain are positive.
- Constitutive-model pressures are reported in MPa.
- Editable elastic moduli are generally given in GPa and converted internally to MPa.
- `N` denotes the number of water molecules per reference structure.
- Dry and water-containing models are independent and do not import files from one another.

## Reproducibility notes

The parameter files are the primary user-editable inputs. Keep each configuration file beside its matching model script. The committed adsorption-stress tables and figures provide reference outputs for the current parameters. Numerical results may show small platform-dependent differences because nonlinear equations and ODEs are solved numerically.
