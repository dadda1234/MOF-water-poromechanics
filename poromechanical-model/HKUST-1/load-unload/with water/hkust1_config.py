"""HKUST-1 parameters. Edit this file, then run hkust1_model.py.

Pressure is positive in compression. Names include the input units.
Only output/N=<N>/stress-strain.png is written by the model.
N is the number of water molecules per reference pore volume.
"""

# 1. Run and output settings
N_VALUE = 32.0
P_MAX_MPA = 757.5
N_STEPS = 1601
SHOW_FIGURE = False
OUTPUT_DIR = "output"

# 2. Reference geometry and temperature
V0_M3 = 4.8068e-27
V0_A3 = V0_M3 / 1.0e-30
T_K = 300.0
POROSITY = 0.5481
WATER_MASS_KG = 3.0e-26

# 3. Open-pore elasticity and survival probability
K_OP_MPA = 12260.0
A1 = 6.80
ETA1_MEV = 76.8
S_CR = 0.725
E_MU_MEV_A3 = 1134.47
# One-way survival: a2=0; Q is the bare-framework energy weight.
# Sm=exp[-r*L(e)], r=exp[-(N/V0)*E_mu/(kBT)], sets the onset only.
# The negative exponent follows the approved barrier-increase convention.

# 4. Adsorption free energy
E0_MEV = 31.64
K_E = 4.40
C = 8.62
N_REF_PER_A3 = 0.0374468
MEV_PER_A3_TO_MPA = 160.22
# Omega(n,e)=integral_0^e pa(n,u)du; Omega(n,0)=0; k0=0.

# 5. Reference strain
HF_DRY = 0.5024002490520032
BETA = 1.2738051222566762
V_W_A3 = 20.0
# hf=HF_DRY-(N/V0_A3)*BETA*V_W_A3; epsilon_fe=epsilon_fs+hf.
# Calibrated to reference strains at N=4,8,...,36 with S_CR=0.725.
# H=epsilon_fe-p_fe/Kcp is distinct from hf.

# 6. Closed-pore modulus
K0_ATM = 105502.00980822474
K0_MPA = K0_ATM * 0.101325
RHO0_KG_M3 = 1289.496
ALPHA_H = 1.5125651186677376
TAIT_M = 1.66
# rho_w=N*mw/[V0*(POROSITY-epsilon_fe)].
# Kcp=K0*[1+ALPHA_H/TAIT_M*((rho_w/rho0)**TAIT_M-1)].
# K0 and ALPHA_H were refitted to N=4,8,...,36 data; not manuscript defaults.
# The water-density law does not specify the dry modulus at N=0.

# 7. Transformation pressures, hardening and manual overrides
P_FE_MPA = 757.5
P_RS_MPA = 0.0
P_RE_MPA = 0.0
LOG_CUTOFF = 6.0
P_FS_OVERRIDE_MPA = None
K_CP_OVERRIDE_MPA = None
H_OVERRIDE = None
# None enables the automatic relation. Manual pressures/moduli here use MPa.
# Logarithmic hardening uses delta=exp(-LOG_CUTOFF), not 1e-6.
# prs=pre=0 closes the five-equation calibration only.
# Reverse transformation is disabled: unloading retains the phase fraction.
# The actual completion fraction is 1-delta, not exactly 1.

# 8. Numerical tolerances and physical constants
ODE_RTOL = 2.0e-10
ODE_ATOL = 2.0e-11
DRY_STRAIN_LIMIT = 0.20
KB_MEV_PER_K = 0.08617333262145
