"""MIL-53(Cr): two-way survival, quadratic hardening, reversible unloading.

Keep this file beside mil53_config.py.
Run: python mil53_model.py [--n 5] [--show]
Install: python -m pip install numpy scipy matplotlib

Wop=Kop*Q(e)*e^2/2+Omega(n,e); pressure=dWop/de.
Total strain=(1-xi)*e+xi*p/Kcp+H*xi; compression is positive.
Q weights energy; hydrated Sm sets onset. Omega(n,0)=0.
Q and Sm are strain-state functions, not backward time integration.
Only the initial stable open-phase branch is inverted; no root jumping.
Pure closed phase may exceed the virtual open-phase pressure range.
Constant-pressure jumps retain both endpoint states; unload fully to zero.
Direction-dependent hardening is phenomenological, not a single-potential proof.
Output: one PNG at output/N=<N>/stress-strain.png. No data files.
CLI pressures use MPa; --k-cp uses GPa in both material packages.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc

import mil53_config as cfg

GPA_TO_MPA = 1000.0
G_CM3_TO_KG_M3 = 1000.0
DA_TO_KG = 1.66053906660e-27


# 1. State and model configuration
@dataclass(frozen=True)
class OpenConfig:
    N: float
    K_op: float
    a1: float
    a2: float
    eta1: float
    eta2: float
    S_cr: float
    E_mu: float
    V0: float = cfg.V0_A3
    T: float = cfg.T_K
    E0: float = cfg.E0_MEV
    kE: float = cfg.K_E
    C: float = cfg.C
    n_ref: float = cfg.N_REF_PER_A3
    adsorption_scale: float = cfg.MEV_PER_A3_TO_MPA
    dry_limit: float = cfg.DRY_STRAIN_LIMIT


# 2. Constitutive model and phase evolution
class OpenPore:
    def __init__(self, c: OpenConfig):
        self.c = c
        vals = np.array(list(asdict(c).values()), dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError("All open-phase parameters must be finite.")
        if min(c.K_op, c.V0, c.T, c.C, c.n_ref, c.dry_limit) <= 0:
            raise ValueError("Kop,V0,T,C,n_ref,dry_limit must be positive.")
        if min(c.N, c.a1, c.a2, c.E0, c.adsorption_scale) < 0 or not 0 < c.S_cr < 1:
            raise ValueError("Require nonnegative N/rates/adsorption and 0<S_cr<1.")
        self.n = c.N/c.V0
        if self.n >= c.n_ref:
            raise ValueError("Require N/V0 < n_ref.")
        self.g1 = c.eta1/(cfg.KB_MEV_PER_K*c.T)
        self.g2 = c.eta2/(cfg.KB_MEV_PER_K*c.T)
        self.rate_factor = float(np.exp(-self.n*c.E_mu/(cfg.KB_MEV_PER_K*c.T)))
        self.e_limit = ((1-self.n/c.n_ref)/c.C*(1-1e-8)
                        if self.n > 0 else c.dry_limit)

        def rhs(e, y):
            return [self.q_prime(e, y[0]), float(self.pa(e)),
                    self.rate_factor*self.q_prime(e, y[2])]

        self.solution = solve_ivp(rhs, (0, self.e_limit), [1., 0., 1.],
            method="DOP853", dense_output=True, rtol=cfg.ODE_RTOL,
            atol=cfg.ODE_ATOL, max_step=self.e_limit/500)
        if not self.solution.success:
            raise RuntimeError(self.solution.message)
        es = np.linspace(0, self.e_limit, 16385)
        prob = self.solution.sol(es)[[0, 2]]
        if not np.all(np.isfinite(prob)) or np.min(prob) < -1e-9 or np.max(prob) > 1+1e-9:
            raise ValueError("Survival probability left [0,1].")
        ps = self.stress(es)
        if not np.all(np.isfinite(ps)):
            raise ValueError("Nonfinite open-phase stress.")
        turn = np.flatnonzero(np.diff(ps) <= 0)
        self.branch_e_max = float(es[turn[0]]) if len(turn) else self.e_limit
        self.p_min = float(ps[0])
        self.p_max = float(self.stress(self.branch_e_max))
        if not self.p_min <= 0 < self.p_max:
            raise ValueError(f"p=0 is not on the initial stable e>=0 branch "
                             f"[{self.p_min:g},{self.p_max:g}] MPa.")
        self.e_zero = self.solve_e(0.)

    def q_prime(self, e, q):
        e = np.asarray(e)
        return (-self.c.a1*np.exp(self.g1*e)*q
                + self.c.a2*np.exp(-self.g2*e)*(1-q))

    def pa(self, e):
        e = np.asarray(e, dtype=float)
        c = self.c
        if self.n == 0 or c.adsorption_scale == 0:
            return np.zeros_like(e)
        d = 1-c.C*e
        n0 = c.n_ref*d
        if np.any(n0 <= self.n):
            raise ValueError("Adsorption domain exceeded: n<n_ref*(1-C*e).")
        a = np.log(n0/self.n)
        e_ads = c.E0*np.exp(c.kE*e)
        return c.adsorption_scale*e_ads*n0*(
            -c.kE*gamma(1.5)*gammaincc(1.5, a)
            + c.C/(2*d)*gamma(.5)*gammaincc(.5, a))

    def fields(self, e):
        if np.any(np.asarray(e) < 0) or np.any(np.asarray(e) > self.e_limit):
            raise ValueError("Strain outside the integrated domain.")
        return self.solution.sol(e)  # Q, Omega, Sm

    def stress(self, e):
        q, _, _ = self.fields(e)
        e = np.asarray(e)
        return self.c.K_op*(q*e+.5*self.q_prime(e, q)*e**2)+self.pa(e)

    def energy(self, e):
        q, omega, _ = self.fields(e)
        return .5*self.c.K_op*q*np.asarray(e)**2+omega

    def solve_e(self, p):
        if not self.p_min-1e-9 <= p <= self.p_max+1e-9:
            raise ValueError(f"p={p:g} MPa outside initial stable op branch "
                             f"[{self.p_min:g},{self.p_max:g}] MPa; no extrapolation.")
        if abs(p-self.p_min) < 1e-10:
            return 0.
        if abs(p-self.p_max) < 1e-10:
            return self.branch_e_max
        return brentq(lambda e: float(self.stress(e)-p), 0, self.branch_e_max,
                      xtol=2e-14, rtol=2e-14)

    def critical_strain(self):
        es = np.linspace(0, self.branch_e_max, 4097)
        below = np.flatnonzero(self.fields(es)[2] <= self.c.S_cr)
        if not len(below):
            raise ValueError("Sm does not reach S_cr before the open-phase stability limit. "
                             "Review the parameter set; do not select a later unstable root.")
        j = int(below[0])
        return brentq(lambda e: float(self.fields(e)[2]-self.c.S_cr),
                      es[j-1], es[j], xtol=1e-14)


@dataclass(frozen=True)
class CycleConfig:
    K_cp: float
    H: float
    p_fs: float
    p_fe: float
    p_rs: float
    p_re: float
    p_max: float = cfg.P_MAX_MPA
    n_steps: int = cfg.N_STEPS
    plateau_rule: str = cfg.PLATEAU_RULE


@dataclass(frozen=True)
class Hardening:
    b1: float
    b2: float
    mu1: float
    mu2: float
    Y: float

    def derivative(self, xi, forward=True):
        return ((self.b1 if forward else self.b2)*np.asarray(xi)
                + self.mu1 + (self.mu2 if forward else -self.mu2))

    def energy(self, xi, forward=True):
        return (.5*(self.b1 if forward else self.b2)*np.asarray(xi)**2
                + (self.mu1+(self.mu2 if forward else -self.mu2))*np.asarray(xi))


class Model:
    def __init__(self, op: OpenPore, c: CycleConfig):
        self.op, self.c = op, c
        numeric = [v for v in asdict(c).values() if not isinstance(v, str)]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Cycle parameters must be finite.")
        if c.K_cp <= 0 or c.H < 0 or c.p_max <= 0 or c.n_steps < 3:
            raise ValueError("Require Kcp,p_max>0, H>=0, n_steps>=3.")
        if not 0 <= c.p_re <= c.p_rs <= c.p_fs <= c.p_fe or c.p_fs == 0:
            raise ValueError("Require 0<=p_re<=p_rs<=p_fs<=p_fe, p_fs>0.")
        if c.plateau_rule != "complete_at_threshold":
            raise ValueError("Supported plateau rule: complete_at_threshold.")
        if c.p_fe > op.p_max:
            raise ValueError("The forward completion state requires an op root beyond its "
                             "stable branch. Check thresholds or specify an instability model.")
        # dI/dp = H+p/Kcp-e_op(p). Check only where phase transformation is used.
        ps = np.linspace(c.p_re, c.p_fe, 301)
        slopes = np.array([c.H+p/c.K_cp-op.solve_e(p) for p in ps])
        if min(slopes) <= 0:
            raise ValueError("I(p) is not strictly increasing in the transformation interval; "
                             "pressure-direction phase selection is not justified.")
        self.min_force_slope = float(min(slopes))
        A, B, C, D = [self.force(p) for p in (c.p_fs, c.p_fe, c.p_rs, c.p_re)]
        b1, b2 = B-A, C-D
        mu2 = (b2-b1)/4
        self.hardening = Hardening(b1, b2, (A+D)/2, mu2, (A-D)/2-mu2)
        if b1 < 0 or b2 < 0 or self.hardening.Y < 0:
            raise ValueError("Calibration produced b1<0, b2<0 or Y<0; review critical pressures.")
        h = self.hardening
        self.calibration_residuals = [A-h.derivative(0)-h.Y, B-h.derivative(1)-h.Y,
            C-h.derivative(1, False)+h.Y, D-h.derivative(0, False)+h.Y,
            h.energy(1)-h.energy(1, False)]
        if max(abs(float(r)) for r in self.calibration_residuals) > 1e-7:
            raise RuntimeError("Five-equation calibration check failed.")

    def force(self, p, e=None):
        e = self.op.solve_e(p) if e is None else e
        return float(self.op.energy(e)-p*e+p*p/(2*self.c.K_cp)+p*self.c.H)

    def update_xi(self, p, old, forward):
        c, h = self.c, self.hardening
        if forward:
            if old == 1:
                return old, "closed_pore"
            if p < c.p_fs-1e-9:
                return old, "forward_stick"
            if h.b1 == 0:
                return 1., "forward_plateau_jump"
            if p >= c.p_fe-1e-9:
                return 1., "forward_complete" if old < 1 else "closed_pore"
            target = (self.force(p)-h.mu1-h.mu2-h.Y)/h.b1
            xi = float(np.clip(max(old, target), 0, 1))
            return xi, "forward" if xi > old else "forward_stick"
        if old == 0:
            return 0., "open_pore"
        if p > c.p_rs+1e-9:
            return old, "reverse_stick"
        if h.b2 == 0:
            return 0., "reverse_plateau_jump"
        if p <= c.p_re+1e-9:
            return 0., "reverse_complete"
        target = (self.force(p)-h.mu1+h.mu2+h.Y)/h.b2
        xi = float(np.clip(min(old, target), 0, 1))
        return xi, "reverse" if xi < old else "reverse_stick"

    def row(self, p, xi, old, forward, state):
        e = self.op.solve_e(p) if p <= self.op.p_max else None
        if e is None and xi != 1:
            raise ValueError("An open-pore fraction remains but its stable strain cannot be inverted.")
        strain = xi*(p/self.c.K_cp+self.c.H)+(1-xi)*(0 if e is None else e)
        q, omega, sm = (map(float, self.op.fields(e)) if e is not None else (None, None, None))
        pa = float(self.op.pa(e)) if e is not None else None
        force = self.force(p, e) if e is not None else None
        h = self.hardening
        pi_f = force-float(h.derivative(xi)) if force is not None else None
        pi_r = force-float(h.derivative(xi, False)) if force is not None else None
        active = pi_f if forward else pi_r
        dx = xi-old
        dissipation = 0. if dx == 0 else active*dx
        if dissipation < -1e-7:
            raise ValueError("Negative phase-transformation dissipation.")
        return dict(leg="loading" if forward else "unloading", pressure=p,
            epsilon=strain, strain_absolute=strain, strain_relative=strain-self.op.e_zero,
            xi=xi, epsilon_t=self.c.H*xi, e_op=e, e_cp=p/self.c.K_cp,
            Q=q, Sm=sm, adsorption_stress=pa, skeleton_pressure=None if pa is None else p-pa,
            adsorption_energy=omega, W_op=None if e is None else float(self.op.energy(e)),
            mechanical_force=force, pi_forward=pi_f, pi_reverse=pi_r, state=state,
            delta_xi=dx, phase_dissipation_endpoint=dissipation,
            op_stress_residual=None if e is None else float(self.op.stress(e)-p))

    def simulate(self):
        c = self.c
        events = [p for p in (c.p_fs, c.p_fe, c.p_rs, c.p_re) if p <= c.p_max]
        load_p = np.unique(np.r_[np.linspace(0, c.p_max, c.n_steps), events])
        if self.op.p_min < 0:
            count = max(3, int(np.ceil(-self.op.p_min/c.p_max*(c.n_steps-1)))+1)
            load_p = np.r_[np.linspace(self.op.p_min, 0, count)[:-1], load_p]
        rows, xi = [], 0.
        for p in load_p:
            old = xi
            xi, state = self.update_xi(float(p), old, True)
            if state == "forward_plateau_jump":
                rows.append(self.row(float(p), old, old, True, "before_forward_jump"))
            rows.append(self.row(float(p), xi, old, True, state))
        peak_xi = xi
        reverse_onset = None
        if xi > 0:
            # At partial loading the reverse onset depends on the attained xi.
            fn = lambda p: self.force(p)-float(self.hardening.derivative(xi, False))+self.hardening.Y
            upper = min(c.p_max, c.p_rs)
            if fn(upper) < -1e-8:
                raise ValueError("Partial-loop reversal is already active at the turning point; "
                                 "the branch-hardening model needs a specified minor-loop rule.")
            if abs(fn(upper)) <= 1e-8:
                reverse_onset = float(upper)
            elif abs(fn(c.p_re)) <= 1e-8:
                reverse_onset = float(c.p_re)
            else:
                reverse_onset = float(brentq(fn, c.p_re, upper, xtol=1e-10))
        unload_p = np.unique(np.r_[np.linspace(0, c.p_max, c.n_steps), events,
                                   [] if reverse_onset is None else [reverse_onset]])[::-1]
        for p in unload_p:
            old = xi
            xi, state = self.update_xi(float(p), old, False)
            if state == "reverse_plateau_jump":
                rows.append(self.row(float(p), old, old, False, "before_reverse_jump"))
            rows.append(self.row(float(p), xi, old, False, state))
        self.events = dict(forward_onset=c.p_fs if c.p_max >= c.p_fs else None,
            forward_completion=c.p_fe if c.p_max >= c.p_fe else None,
            reverse_onset=reverse_onset, reverse_completion=c.p_re if peak_xi > 0 else None)
        return rows


# 3. Adsorption-dependent material parameters
def dry_critical_probability():
    """Compute Q(eps_cr_dry) using the same bare two-state ODE as the user script."""
    if not np.isfinite(cfg.EPS_CR_DRY) or cfg.EPS_CR_DRY <= 0 or cfg.T_K <= 0:
        raise ValueError("EPS_CR_DRY and T_K must be positive and finite.")
    if min(cfg.A1, cfg.A2) < 0:
        raise ValueError("Survival rates must be nonnegative.")
    kbt = cfg.KB_MEV_PER_K*cfg.T_K

    def rhs(e, y):
        return [-cfg.A1*np.exp(cfg.ETA1_MEV*e/kbt)*y[0]
                + cfg.A2*np.exp(-cfg.ETA2_MEV*e/kbt)*(1-y[0])]

    solution = solve_ivp(rhs, (0., cfg.EPS_CR_DRY), [1.], method="DOP853",
        rtol=cfg.ODE_RTOL, atol=cfg.ODE_ATOL, max_step=cfg.EPS_CR_DRY/200)
    if not solution.success:
        raise RuntimeError(solution.message)
    scr = float(solution.y[0, -1])
    if not 0 < scr < 1:
        raise ValueError("Dry critical probability must lie strictly between zero and one.")
    return scr


def open_config(N=None):
    N = cfg.N_VALUE if N is None else float(N)
    return OpenConfig(N=N, K_op=cfg.K_OP_GPA*GPA_TO_MPA, a1=cfg.A1, a2=cfg.A2,
        eta1=cfg.ETA1_MEV, eta2=cfg.ETA2_MEV, S_cr=dry_critical_probability(),
        E_mu=cfg.E_MU_MEV_A3)


def material_parameters(N=None, *, p_fs=None, p_fe=None, p_rs=None,
                        p_re=None, K_cp=None, H=None, p_max=None, n_steps=None):
    """Function arguments use internal MPa; editable config moduli use GPa."""
    op = OpenPore(open_config(N))
    N, n = op.c.N, op.n
    def selected(value, default):
        return default if value is None else float(value)
    fs = selected(p_fs, cfg.P_FS_OVERRIDE_MPA)
    e_fs = op.critical_strain() if fs is None else op.solve_e(fs)
    fs = float(op.stress(e_fs)) if fs is None else fs
    fe = selected(p_fe, cfg.P_FE_OVERRIDE_MPA)
    fe = fs if fe is None else fe
    hf = cfg.HF_DRY-n*cfg.BETA*cfg.V_W_A3
    e_fe = e_fs+hf
    cp_override = (None if cfg.K_CP_OVERRIDE_GPA is None
                   else cfg.K_CP_OVERRIDE_GPA*GPA_TO_MPA)
    cp = selected(K_cp, cp_override)
    rho, cp_law = None, None
    if N > 0:
        if not 0 < e_fe < cfg.POROSITY or hf <= 0:
            raise ValueError("Reference-strain geometry invalid: need hf>0 and 0<epsilon_fe<porosity.")
        rho = N*(cfg.WATER_MASS_DA*DA_TO_KG)/(op.c.V0*1e-30*(cfg.POROSITY-e_fe))
        rho0_si = cfg.RHO0_G_CM3*G_CM3_TO_KG_M3
        cp_law = (cfg.K0_GPA*GPA_TO_MPA)*(1+cfg.ALPHA_H/cfg.TAIT_M*((rho/rho0_si)**cfg.TAIT_M-1))
        cp = cp_law if cp is None else cp
    H_value = selected(H, cfg.H_OVERRIDE)
    if cp is not None:
        if cp <= 0:
            raise ValueError("Kcp from parameters is nonpositive; review the density law.")
        H_value = e_fe-fe/cp if H_value is None else H_value
    rs = selected(p_rs, cfg.P_RS_OVERRIDE_MPA)
    gap = n*cfg.GEOMETRY_GAMMA*cfg.H1_A*cfg.V_W_A3
    disjoining = (cfg.PI0_GPA*GPA_TO_MPA*np.cos(cfg.OMEGA_PER_A*gap+cfg.PHI_RAD)*np.exp(-cfg.KAPPA_PER_A*gap)
                  if N > 0 else 0.)
    if rs is None and cfg.P_RS_DRY_MPA is not None:
        rs = cfg.P_RS_DRY_MPA+disjoining
    re = selected(p_re, cfg.P_RE_OVERRIDE_MPA)
    re = rs if re is None else re
    missing = []
    if cp is None:
        missing.append("K_CP_OVERRIDE_GPA: dry (N=0) closed-pore modulus, GPa")
    if rs is None:
        missing.append("P_RS_OVERRIDE_MPA or P_RS_DRY_MPA: reverse onset pressure")
    report = dict(N=N, n_per_A3=n, parameter_set="user_reference_scripts",
        input_units=dict(modulus="GPa", critical_pressure="MPa", Pi0="GPa",
            energy="meV", E_mu="meV*A^3", reference_density="g/cm^3", molecular_mass="Da"),
        K_op_GPa=cfg.K_OP_GPA, K_cp_GPa=None if cp is None else cp/GPA_TO_MPA,
        E_mu_meV_A3=op.c.E_mu, S_cr=op.c.S_cr, dry_critical_strain=cfg.EPS_CR_DRY,
        equivalent_gamma_per_molecule=op.c.E_mu/(op.c.V0*cfg.KB_MEV_PER_K*op.c.T),
        K_op_MPa=op.c.K_op, K_cp_MPa=cp, K_cp_law_MPa=cp_law,
        H=H_value, p_fs=fs, p_fe=fe, p_rs=rs, p_re=re,
        critical_strain=e_fs, reference_completion_strain=e_fe, hf=hf,
        reference_strain_parameters=dict(hf_dry=cfg.HF_DRY,beta=cfg.BETA,
            water_volume_A3=cfg.V_W_A3,calibration_N_range=[2.,5.]),
        reference_strain_is_extrapolation=bool(N<2 or N>5),
        confined_water_density_kg_m3=rho, pore_gap_A=gap,
        disjoining_pressure_MPa=float(disjoining),
        zero_pressure_strain=op.e_zero, pressure_at_zero_strain=op.p_min,
        op_stable_pressure_limit_MPa=op.p_max, Sm_at_onset=float(op.fields(e_fs)[2]),
        rate_factor=op.rate_factor, missing_inputs=missing,
        assumptions=["Q is bare-framework survival; Sm sets the onset only.",
            "Both hydrated rates use exp(-n*E_mu/kBT), correcting the manuscript sign.",
            "Omega(n,0)=0; no absolute phase adsorption-energy offset was supplied.",
            "rho is absorbed into b1,b2; xi endpoints are exactly 0 and 1.",
            "At equal start/end pressures xi jumps at the threshold by an explicit selection rule.",
            "Full unloading reverses xi; Q is a state function, not a separately reversed time ODE.",
            "The user reference-script parameters replace the old article/fitted presets.",
            "S_cr is recomputed as bare Q(EPS_CR_DRY), not held at a rounded constant.",
            "The reference-script K=318293.92 is interpreted as atm; K_op=32.251131444 GPa.",
            "Editable config uses article units; internal pressure/modulus and plotted pressure use MPa.",
            "Reference-strain HF_DRY and BETA were refitted to N=2,3,4,5 targets, not interpolated exactly.",
            "Reference-strain use outside N=2..5 is extrapolation and can violate constitutive admissibility.",
            "Closed-pore K0, alpha_H and m were refitted to the latest third-stage moduli at N=0.5..5; RMSE=2.26875861 GPa.",
            "Rounded manuscript disjoining parameters are provisional."])
    if missing:
        return op, None, report
    cycle = CycleConfig(float(cp), float(H_value), float(fs), float(fe), float(rs), float(re),
        p_max=cfg.P_MAX_MPA if p_max is None else float(p_max),
        n_steps=cfg.N_STEPS if n_steps is None else int(n_steps))
    return op, cycle, report


# 4. Numerical verification
def diagnostics(model, rows):
    load = [r for r in rows if r["leg"] == "loading"]
    unload = [r for r in rows if r["leg"] == "unloading"]
    h = model.hardening
    active = []
    for r in rows:
        if r["delta_xi"] != 0:
            active.append(abs((r["pi_forward"]-h.Y) if r["delta_xi"] > 0 else (r["pi_reverse"]+h.Y)))
    checks = dict(loading_end_strain=load[-1]["epsilon"], unloading_zero_strain=unload[-1]["epsilon"],
        initial_zero_pressure_strain=model.op.e_zero, peak_xi=load[-1]["xi"], final_xi=unload[-1]["xi"],
        max_calibration_residual=float(max(abs(v) for v in model.calibration_residuals)),
        max_stress_residual=max(abs(r["op_stress_residual"]) for r in rows if r["op_stress_residual"] is not None),
        max_active_residual=max(active, default=0.),
        loading_xi_nondecreasing=bool(np.all(np.diff([r["xi"] for r in load]) >= -1e-12)),
        unloading_xi_nonincreasing=bool(np.all(np.diff([r["xi"] for r in unload]) <= 1e-12)),
        minimum_phase_dissipation=min(r["phase_dissipation_endpoint"] for r in rows),
        minimum_dI_dp=model.min_force_slope, actual_events_MPa=model.events)
    if (checks["max_stress_residual"] > 1e-5 or checks["max_active_residual"] > 1e-5
            or not checks["loading_xi_nondecreasing"] or not checks["unloading_xi_nonincreasing"]):
        raise RuntimeError(f"Cycle verification failed: {checks}")
    return checks



# 5. Figure export
def export_plot(model, rows, report, out, show):
    """Save one minimal, deterministic stress-strain figure (PNG only).

    Single quantitative panel; no experimental uncertainty is implied.
    Retain the full unloading branch and all constant-pressure jump points.
    Show nonnegative pressure and absolute strain without shifting the origin.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    for leg, label, color, style in (
        ("loading", "Loading", "#20639B", "-"),
        ("unloading", "Unloading", "#C86D35", "--"),
    ):
        selected = [r for r in rows if r["leg"] == leg and r["pressure"] >= 0]
        ax.plot(
            [r["pressure"] for r in selected],
            [r["strain_absolute"] for r in selected],
            color=color, ls=style, lw=1.8, label=label,
        )
    N = report["N"]
    ax.set(
        xlim=(0.0, 1.025*model.c.p_max),
        ylim=(0.0, max(0.02, max(r["strain_absolute"] for r in rows)*1.10)),
        xlabel="Pressure, p (MPa)",
        ylabel=r"Strain, $\varepsilon$",
        title=f"N={N:g}",
    )
    ax.legend(loc="lower right")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stress-strain.png"
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path

# 6. Command-line entry point
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=float, help="Water molecules per reference volume")
    for name in ("p-max", "p-fs", "p-fe", "p-rs", "p-re"):
        parser.add_argument("--"+name, type=float, help="Pressure in MPa")
    parser.add_argument("--k-cp", type=float, help="Closed-pore modulus in GPa")
    parser.add_argument("--h", type=float, help="Maximum transformation strain")
    parser.add_argument("--inspect", action="store_true", help="Print parameters only; write no files")
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window")
    args = parser.parse_args(argv)
    op, c, report = material_parameters(
        args.n, p_max=args.p_max, p_fs=args.p_fs, p_fe=args.p_fe,
        p_rs=args.p_rs, p_re=args.p_re,
        K_cp=None if args.k_cp is None else args.k_cp*1000, H=args.h,
    )
    if args.inspect:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if c is None:
        parser.error("Missing material input(s): "+"; ".join(report["missing_inputs"]))
    model = Model(op, c)
    rows = model.simulate()
    diagnostics(model, rows)
    N = report["N"]
    out = Path(__file__).resolve().parent / cfg.OUTPUT_DIR / f"N={N:g}"
    path = export_plot(model, rows, report, out, args.show or cfg.SHOW_FIGURE)
    print(f"N={N:g}; Kcp={c.K_cp/1000:.6f} GPa; saved {path}")


if __name__ == "__main__":
    main()
