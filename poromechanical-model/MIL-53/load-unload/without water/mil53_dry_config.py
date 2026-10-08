"""Dry MIL-53(Cr) parameters. Edit this file, then run mil53_dry_model.py.

No water-dependent stress, energy, density or reference-strain relations.
Kcp and H are direct inputs. Output: output/N=0/stress-strain.png only.
"""

# 1. Run and output settings

P_MAX_MPA = 2000.0
N_STEPS = 1201
SHOW_FIGURE = False
OUTPUT_DIR = "output"


# 2. Survival probability and open-pore elasticity

K_OP_GPA = 32.251131444
A1 = 10.7447
A2 = 5.3984
ETA1_MEV = 25.852
ETA2_MEV = 72.46
T_K = 300.0
EPS_CR_DRY = 0.132
S_CR = None
# Q'=-a1*exp(eta1*e/kBT)*Q+a2*exp(-eta2*e/kBT)*(1-Q), Q(0)=1.
# W=Kop*Q(e)*e^2/2; p=Kop*(Q*e+Q_prime*e^2/2).
# EPS_CR_DRY sets the critical strain directly; otherwise solve Q(e)=S_CR.


# 3. Direct closed-pore modulus and transformation strain

K_CP_GPA = 32.0
H = 0.3
# Total strain=(1-xi)*e_op+xi*p/Kcp+H*xi.


# 4. Transformation thresholds and hardening

P_FS_OVERRIDE_MPA = None
P_FE_MPA = None
P_RS_MPA = 5410.0 * 0.101325
P_RE_MPA = None
PLATEAU_RULE = "complete_at_threshold"
# Quadratic hardening; pfe=None means pfe=pfs. Reverse transformation is enabled.
# The original accepted survival parameters are retained, not the separate fit.
# All pressures use MPa. None for P_RE_MPA means p_re=p_rs.
# P_FS_OVERRIDE_MPA=None computes the onset from the survival criterion.


# 5. Numerical tolerances and constants

ODE_RTOL = 2.0e-10
ODE_ATOL = 2.0e-12
DRY_STRAIN_LIMIT = 0.6
KB_MEV_PER_K = 0.08617333262145
