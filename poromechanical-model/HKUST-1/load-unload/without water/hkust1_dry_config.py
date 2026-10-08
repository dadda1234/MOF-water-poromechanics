"""Dry HKUST-1 with the accepted onset continuation. Run hkust1_dry_model.py.

No water-dependent stress, energy, density or reference-strain relations.
Kcp and H are direct inputs. Output: output/N=0/stress-strain.png only.
"""

# 1. Run and output settings

P_MAX_MPA = 6000.0 * 0.101325  # 6000 atm = 607.950 MPa
N_STEPS = 1201
SHOW_FIGURE = False
OUTPUT_DIR = "output"


# 2. Survival probability and open-pore elasticity

K_OP_GPA = 12.26
A1 = 6.80
A2 = 0.0
ETA1_MEV = 76.8
ETA2_MEV = 0.0
T_K = 300.0
EPS_CR_DRY = None
S_CR = 0.725
# Q'=-a1*exp(eta1*e/kBT)*Q+a2*exp(-eta2*e/kBT)*(1-Q), Q(0)=1.
# W=Kop*Q(e)*e^2/2; p=Kop*(Q*e+Q_prime*e^2/2).
# EPS_CR_DRY sets the critical strain directly; otherwise solve Q(e)=S_CR.


# 3. Direct closed-pore modulus and transformation strain

K_CP_GPA = 9.6
H = 0.485
# Total strain=(1-xi)*e_op+xi*p/Kcp+H*xi.


# 4. Transformation thresholds and hardening

P_FS_OVERRIDE_MPA = None
P_FE_MPA = 5000.0 * 0.101325  # 5000 atm = 506.625 MPa
P_RS_MPA = 0.0
P_RE_MPA = None
LOG_CUTOFF = 6.0
# Logarithmic hardening and irreversible unloading: xi is capped at 1-exp(-6).
POST_ONSET_RULE = "fixed_open_reference"
# At Q=S_CR, freeze the open strain and stored energy at their onset values.
# Then calibrate the logarithmic hardening using the effective phase force.
# This is an added phenomenological closure, not continuation of the original
# softening stress law. The remaining open fraction is not declared stress-free.
# Completion still means xi=1-exp(-6); no artificial jump to xi=1 is inserted.
# On unloading below pfs, the residual open fraction follows the elastic branch.
# All pressures use MPa. None for P_RE_MPA means p_re=p_rs.
# P_FS_OVERRIDE_MPA=None computes the onset from the survival criterion.


# 5. Numerical tolerances and constants

ODE_RTOL = 2.0e-10
ODE_ATOL = 2.0e-12
DRY_STRAIN_LIMIT = 0.20
KB_MEV_PER_K = 0.08617333262145
