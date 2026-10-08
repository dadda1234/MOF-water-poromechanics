"""MIL-53(Cr) parameters. Edit this file, then run mil53_model.py.

Pressure is positive in compression. Names include the input units.
Only output/N=<N>/stress-strain.png is written by the model.
N is the number of water molecules per reference volume.
"""

# 1. Run and output settings
N_VALUE = 5.0
P_MAX_MPA = 2000.0
N_STEPS = 1201
SHOW_FIGURE = False
OUTPUT_DIR = "output"

# 2. Reference geometry and temperature
V0_A3 = 793.65
T_K = 300.0
POROSITY = 0.50
WATER_MASS_DA = 3.0e-26 / 1.66053906660e-27

# 3. Open-pore elasticity and survival probability
K_OP_GPA = 32.251131444
A1 = 10.7447
A2 = 5.3984
ETA1_MEV = 25.852
ETA2_MEV = 72.46
EPS_CR_DRY = 0.132
E_MU_MEV_A3 = 2212.395515357261
# Two-way survival: Q'=-a1*exp(g1*e)*Q+a2*exp(-g2*e)*(1-Q), Q(0)=1.
# Sm uses the same two rates multiplied by exp[-(N/V0)*E_mu/(kBT)].
# Q weights the elastic energy; Sm determines the onset only.
# S_cr=Q(EPS_CR_DRY) is recomputed, not held at a rounded probability.
# Kop retains the conversion of the supplied 318293.92 value from atm.
# The separate survival-only fit has NOT been adopted.

# 4. Adsorption free energy
E0_MEV = 240.98
K_E = 0.625
C = 2.55
N_REF_PER_A3 = 0.0315
MEV_PER_A3_TO_MPA = 160.22
# Omega(n,e)=integral_0^e pa(n,u)du; Omega(n,0)=0; k0=0.
# K_E retains the unrounded value, rather than the table value 0.63.

# 5. Reference strain
HF_DRY = 0.29533125358525900
BETA = 1.4304114058365884
V_W_A3 = 20.0
# hf=HF_DRY-(N/V0_A3)*BETA*V_W_A3; epsilon_fe=epsilon_fs+hf.
# Fitted to reference strains at N=2,3,4,5; use outside that range is extrapolation.
# H=epsilon_fe-p_fe/Kcp is distinct from hf.

# 6. Closed-pore modulus
K0_GPA = 50.8136812801
RHO0_G_CM3 = 1.02
ALPHA_H = 3.0751832185
TAIT_M = 5.2325489587
# rho_w=N*mw/[V0*(POROSITY-epsilon_fe)].
# Kcp=K0*[1+ALPHA_H/TAIT_M*((rho_w/rho0)**TAIT_M-1)].
# Fitted at N=0.5,1,...,5: RMSE=2.26875861 GPa.
# N=0 needs an independently supplied K_CP_OVERRIDE_GPA.

# 7. Transformation pressures, hardening and manual overrides
P_FS_OVERRIDE_MPA = None
P_FE_OVERRIDE_MPA = None
P_RS_OVERRIDE_MPA = None
P_RE_OVERRIDE_MPA = None
K_CP_OVERRIDE_GPA = None
H_OVERRIDE = None
P_RS_DRY_MPA = 5410.0 * 0.101325
PI0_GPA = 2.53
OMEGA_PER_A = 0.10
PHI_RAD = -1.57
KAPPA_PER_A = 0.38
GEOMETRY_GAMMA = 5.19
H1_A = 24.44
# None enables automatic relations; pressures use MPa, modulus override uses GPa.
# pfe defaults to pfs, and pre defaults to prs: constant-pressure transitions.
# h=(N/V0_A3)*GEOMETRY_GAMMA*H1_A*V_W_A3.
# prs=P_RS_DRY_MPA+1000*PI0_GPA*cos(OMEGA_PER_A*h+PHI_RAD)*exp(-KAPPA_PER_A*h).
# At N=0 use the dry baseline directly, avoiding the rounded-phase offset.
# Quadratic branch hardening; exact phase-fraction endpoints 0 and 1.
# Reverse transformation is enabled; unload fully to zero pressure.

# 8. Numerical tolerances and physical constants
ODE_RTOL = 2.0e-10
ODE_ATOL = 2.0e-12
DRY_STRAIN_LIMIT = 0.6
PLATEAU_RULE = "complete_at_threshold"
KB_MEV_PER_K = 0.08617333262145
# Zero hardening slope: complete the phase-fraction jump at the threshold.
