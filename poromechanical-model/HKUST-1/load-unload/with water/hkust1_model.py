"""HKUST-1: one-way survival, logarithmic hardening, irreversible unloading.

Keep this file beside hkust1_config.py.
Run: python hkust1_model.py [--n 32] [--show]
Install: python -m pip install numpy scipy matplotlib

Wop=Kop*Q(e)*e^2/2+Omega(n,e); pressure=dWop/de.
Total strain=(1-xi)*e+xi*p/Kcp+H*xi; compression is positive.
Q weights energy; hydrated Sm sets onset. Omega(n,0)=0.
Only the initial stable open-phase branch is inverted.
The five-equation calibration retains the prescribed branch hardening.
Unloading retains xi, with a finite upper cutoff xi=1-exp(-6).
Output: one PNG at output/N=<N>/stress-strain.png. No data files.
CLI pressures use MPa; --k-cp uses GPa in both material packages.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc

from hkust1_config import (
    N_VALUE,
    V0_M3,
    V0_A3,
    K_OP_MPA,
    P_FE_MPA,
    P_RS_MPA,
    P_RE_MPA,
    P_MAX_MPA,
    LOG_CUTOFF,
    N_STEPS,
    SHOW_FIGURE,
    P_FS_OVERRIDE_MPA,
    K_CP_OVERRIDE_MPA,
    H_OVERRIDE,
    E0_MEV,
    K_E,
    C,
    N_REF_PER_A3,
    MEV_PER_A3_TO_MPA,
    A1,
    ETA1_MEV,
    T_K,
    S_CR,
    E_MU_MEV_A3,
    K0_ATM,
    K0_MPA,
    RHO0_KG_M3,
    ALPHA_H,
    TAIT_M,
    POROSITY,
    WATER_MASS_KG,
    HF_DRY,
    BETA,
    V_W_A3,
    OUTPUT_DIR,
    ODE_RTOL,
    ODE_ATOL,
    DRY_STRAIN_LIMIT,
    KB_MEV_PER_K,
)

# 1. State and model configuration
@dataclass(frozen=True)
class Config:
    """Derived cycle configuration; edit user inputs in hkust1_config.py."""
    K_op: float
    K_cp: float
    H: float
    n: float
    E0: float = E0_MEV
    kE: float = K_E
    k0: float = 0.0
    C: float = C
    n_ref: float = N_REF_PER_A3
    adsorption_scale: float = MEV_PER_A3_TO_MPA
    use_Q: bool = True
    a1: float = A1
    a2: float = 0.0
    eta1_meV: float = ETA1_MEV
    eta2_meV: float = 0.0
    T_K: float = T_K
    p_fs: float = 0.0
    p_fe: float = P_FE_MPA
    p_rs: float = P_RS_MPA
    p_re: float = P_RE_MPA
    delta: float = float(np.exp(-LOG_CUTOFF))
    p_max: float = P_MAX_MPA
    n_steps: int = N_STEPS
    reverse_enabled: bool = False
    degenerate_reverse_at_equality: str = "hold"
    calibrate_hardening: bool = True
    fixed_hardening: tuple[float, float, float, float, float] | None = None
    pressure_label: str = "Pressure, p (MPa)"
    dry_strain_limit: float = DRY_STRAIN_LIMIT
    ode_rtol: float = ODE_RTOL
    ode_atol: float = ODE_ATOL

@dataclass(frozen=True)
class Hardening:
    b1: float
    b2: float
    mu1: float
    mu2: float
    Y: float

    def forward_derivative(self, xi):
        return self.b1 * (-np.log1p(-np.asarray(xi)) - 2.0) + self.mu1 + self.mu2

    def reverse_derivative(self, xi):
        if abs(self.b2) < 1.0e-12:
            return np.zeros_like(np.asarray(xi), dtype=float) + self.mu1 - self.mu2
        return self.b2 * (np.log(np.asarray(xi)) + 1.0) + self.mu1 - self.mu2


# 2. Constitutive model and phase evolution
class Model:
    def __init__(self, config: Config):
        self.c = config
        c = config
        if not (c.K_op > 0 and c.K_cp > 0 and c.H >= 0 and c.T_K > 0):
            raise ValueError("K_op,K_cp,T_K must be positive; H must be nonnegative.")
        if not (0 <= c.n < c.n_ref and c.C > 0 and 0 < c.delta < 0.01):
            raise ValueError("Require 0 <= n < n_ref, C > 0, 0 < delta < .01.")
        if min(c.a1, c.a2, c.E0, c.adsorption_scale) < 0:
            raise ValueError("Rates, E0 and adsorption_scale must be nonnegative.")
        if not (0 <= c.p_fs < c.p_fe and 0 <= c.p_re <= c.p_rs):
            raise ValueError("Require 0 <= p_fs < p_fe and 0 <= p_re <= p_rs.")
        if c.p_max <= 0 or c.n_steps < 3:
            raise ValueError("p_max must be positive and n_steps >= 3.")
        if c.degenerate_reverse_at_equality not in ("hold", "complete"):
            raise ValueError("Degenerate reverse rule must be hold or complete.")
        self.xi_max = 1.0 - c.delta
        self.e_limit = ((1.0 - c.n / c.n_ref) / c.C * (1.0 - 1.0e-8)
                        if c.n > 0 else c.dry_strain_limit)
        self.g1 = c.eta1_meV / (KB_MEV_PER_K * c.T_K)
        self.g2 = c.eta2_meV / (KB_MEV_PER_K * c.T_K)

        def rhs(e, state):
            return [float(self.q_derivative(e, state[0])), float(self.pa(e))]

        self.integral = solve_ivp(
            rhs, (0.0, self.e_limit), [1.0, 0.0], method="DOP853",
            dense_output=True, rtol=c.ode_rtol, atol=c.ode_atol,
            max_step=self.e_limit / 400.0,
        )
        if not self.integral.success:
            raise RuntimeError(f"Numerical Q/adsorption integration failed: {self.integral.message}")

        # Follow only the initial positive-tangent branch. Do not jump to an
        # unrelated root if a new parameter set produces an elastic spinodal.
        scan_e = np.linspace(0.0, self.e_limit, 8193)
        scan_p = self.stress(scan_e)
        if not np.all(np.isfinite(scan_p)):
            raise RuntimeError("Nonfinite open-phase stress in the allowed domain.")
        bad = np.flatnonzero(np.diff(scan_p) <= 0.0)
        self.branch_e_max = float(scan_e[bad[0]]) if len(bad) else self.e_limit
        self.branch_p_min = float(self.stress(0.0))
        self.branch_p_max = float(self.stress(self.branch_e_max))
        required_p = (max(c.p_max, c.p_fs, c.p_fe, c.p_rs, c.p_re)
                      if c.calibrate_hardening else c.p_max)
        if self.stress(0.0) > 0 or required_p > self.branch_p_max:
            raise ValueError(
                f"Initial stable open-phase branch supports p in "
                f"[{self.stress(0.0):.8g}, {self.branch_p_max:.8g}], but "
                f"p=0 and p={required_p:g} are required. Do not clip or invent a "
                "root; check parameters/domain or add a specified instability rule."
            )
        self.e_zero = self.solve_e(0.0)
        self.hardening, self.calibration_residuals = self.calibrate()

    def q_derivative(self, e, q):
        if not self.c.use_Q:
            return np.zeros_like(np.asarray(e), dtype=float)
        A = self.c.a1 * np.exp(self.g1 * np.asarray(e))
        B = self.c.a2 * np.exp(-self.g2 * np.asarray(e))
        return -A * q + B * (1.0 - q)

    def pa(self, e):
        """User-supplied p_a, fixed n; invalid states raise, not sentinel stresses."""
        c = self.c
        e = np.asarray(e, dtype=float)
        if c.n == 0 or c.adsorption_scale == 0:
            return np.zeros_like(e)
        denom = 1.0 - c.C * e
        n0 = c.n_ref * denom
        if np.any(n0 <= c.n):
            raise ValueError("Adsorption domain violated: n < n_ref*(1-C*e) is required.")
        a = np.log(n0 / c.n)
        I_sqrt = n0 * gamma(1.5) * gammaincc(1.5, a)
        I_inv_sqrt = n0 * gamma(0.5) * gammaincc(0.5, a)
        E = c.E0 * np.exp(c.kE * e)
        return c.adsorption_scale * (
            c.n * c.k0 - E * c.kE * I_sqrt
            + E * c.C / (2.0 * denom) * I_inv_sqrt
        )

    def q_omega(self, e):
        e = np.asarray(e, dtype=float)
        if np.any(e < 0) or np.any(e > self.e_limit):
            raise ValueError("Strain is outside the integrated open-phase domain.")
        return self.integral.sol(e)

    def energy(self, e):
        q, omega = self.q_omega(e)
        return 0.5 * self.c.K_op * q * np.asarray(e)**2 + omega

    def stress(self, e):
        q, _ = self.q_omega(e)
        e = np.asarray(e)
        return self.c.K_op * (q * e + 0.5 * e**2 * self.q_derivative(e, q)) + self.pa(e)

    def solve_e(self, p):
        if not self.branch_p_min <= p <= self.branch_p_max:
            raise ValueError(f"Pressure {p:g} is outside the tracked stable branch.")
        return brentq(lambda e: float(self.stress(e) - p), 0.0, self.branch_e_max,
                      xtol=2.0e-14, rtol=2.0e-14)

    def mechanical_force(self, p, e=None):
        e = self.solve_e(p) if e is None else e
        return float(self.energy(e) - p * e + p*p / (2*self.c.K_cp) + p*self.c.H)

    def calibrate(self):
        c = self.c
        ld = np.log(c.delta)
        # Unknown vector: b1,b2,mu1,mu2,Y. Fifth row is f+(1)=f-(1).
        mat = np.array([
            [-2.0, 0.0, 1.0, 1.0, 1.0],
            [-ld-2.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, ld+1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 0.0, 2.0, 0.0],
        ])
        if c.calibrate_hardening:
            rhs = np.array([self.mechanical_force(p) for p in
                            (c.p_fs, c.p_fe, c.p_rs, c.p_re)] + [0.0])
            values = np.linalg.solve(mat, rhs)
        else:
            if c.fixed_hardening is None or len(c.fixed_hardening) != 5:
                raise ValueError("Provide fixed_hardening=(b1,b2,mu1,mu2,Y).")
            values = np.asarray(c.fixed_hardening, dtype=float)
        if min(values[0], values[1]) < -1.0e-9 or values[4] < 0:
            raise ValueError(f"Need b1,b2 >= 0 and Y >= 0; obtained {values}.")
        values[:2] = np.maximum(values[:2], 0.0)
        values[:2][values[:2] < 1e-12] = 0.0
        return Hardening(*map(float, values)), (mat @ values - rhs if c.calibrate_hardening else None)

    def forward_events(self):
        """Actual onset/completion inside this loading range, from pi=Y."""
        h = self.hardening
        result = {}
        for key, xi in (("onset", 0.0), ("completion", self.xi_max)):
            fn = lambda p: self.mechanical_force(p) - float(h.forward_derivative(xi)) - h.Y
            tol = 1e-10 * max(1.0, abs(h.Y), abs(self.mechanical_force(self.c.p_max)))
            if fn(0.0) >= 0:
                result[key] = 0.0
            elif abs(fn(self.c.p_max)) <= tol:
                result[key] = float(self.c.p_max)
            elif fn(self.c.p_max) < 0:
                result[key] = None
            else:
                result[key] = float(brentq(fn, 0.0, self.c.p_max, xtol=1e-9))
        return result

    def update_xi(self, force, old_xi, loading):
        """Rate-independent, history-preserving active-set update."""
        h = self.hardening
        tol = 2.0e-10 * max(1.0, abs(force), h.Y)
        if loading:
            if old_xi >= self.xi_max:
                return old_xi, "upper_cutoff"
            residual = lambda x: force - float(h.forward_derivative(x)) - h.Y
            if residual(old_xi) <= tol:
                return old_xi, "forward_stick"
            if residual(self.xi_max) >= 0:
                return self.xi_max, "upper_cutoff"
            return brentq(residual, old_xi, self.xi_max, xtol=5e-15, rtol=1e-14), "forward"

        if not self.c.reverse_enabled or old_xi == 0:
            return old_xi, "reverse_disabled" if not self.c.reverse_enabled else "initial_phase"
        residual = lambda x: force - float(h.reverse_derivative(x)) + h.Y
        r_old = residual(old_xi)
        if abs(h.b2) < 1.0e-12:
            if r_old < -tol:
                return 0.0, "reverse_jump"
            if abs(r_old) <= tol and self.c.degenerate_reverse_at_equality == "complete":
                return 0.0, "chosen_constant_pressure_completion"
            return old_xi, "degenerate_equality_hold" if abs(r_old) <= tol else "reverse_stick"
        if r_old >= -tol:
            return old_xi, "reverse_stick"
        lo = min(self.c.delta, old_xi)
        if residual(lo) <= 0:
            return 0.0, "lower_cutoff"  # residual trace <= delta is discarded explicitly
        return brentq(residual, lo, old_xi, xtol=5e-15, rtol=1e-14), "reverse"

    def simulate(self):
        c = self.c
        # Include actual event pressures, not fixed reference pressures; onset
        # may therefore move when only n changes.
        events = self.forward_events()
        p_values = np.unique(np.r_[np.linspace(0.0, c.p_max, c.n_steps),
                                  [p for p in events.values() if p is not None],
                                  [p for p in (c.p_rs, c.p_re) if 0 <= p <= c.p_max]])
        # Extend only loading to zero strain. Preserve the original p>=0 grid
        # and the zero-pressure unloading endpoint (no tensile unloading cycle).
        if self.branch_p_min < 0:
            negative_steps = max(3, int(np.ceil(-self.branch_p_min/c.p_max*(c.n_steps-1)))+1)
            negative_p = np.linspace(self.branch_p_min, 0.0, negative_steps)[:-1]
            p_values = np.r_[negative_p, p_values]
        zero_index = int(np.searchsorted(p_values, 0.0))
        e_values = np.array([self.solve_e(float(p)) for p in p_values])
        forces = np.array([self.mechanical_force(p, e) for p, e in zip(p_values, e_values)])
        if np.any(np.diff(forces) < -1e-7):
            raise ValueError("Mechanical driving force is not monotone on this branch; "
                             "pressure-direction branch selection is not justified.")
        if forces[0] - self.hardening.forward_derivative(0.0) > self.hardening.Y + 1e-7:
            raise ValueError("The specified xi=0 initial state is unstable at the loading start.")
        rows = []
        xi = 0.0
        for loading, indices in ((True, range(len(p_values))),
                                 (False, range(len(p_values)-1, zero_index-1, -1))):
            for i in indices:
                p, e, force = float(p_values[i]), float(e_values[i]), float(forces[i])
                old = xi
                xi, state = self.update_xi(force, old, loading)
                q, omega = map(float, self.q_omega(e))
                strain = (1.0-xi)*e + xi*p/c.K_cp + c.H*xi
                pi_f = force - float(self.hardening.forward_derivative(xi))
                pi_r = (force - float(self.hardening.reverse_derivative(xi))
                        if xi > 0 or self.hardening.b2 == 0 else None)
                pi_active = pi_f if loading else pi_r
                dissipation = 0.0 if xi == old else float(pi_active * (xi-old)) if pi_active is not None else None
                rows.append(dict(
                    leg="loading" if loading else "unloading", pressure=p,
                    strain=strain-self.e_zero, strain_absolute=strain, xi=xi,
                    epsilon_t=c.H*xi, e_op=e, e_cp=p/c.K_cp, Q=q,
                    adsorption_stress=float(self.pa(e)), adsorption_energy=omega,
                    W_op=float(self.energy(e)), mechanical_force=force,
                    pi_forward=pi_f, pi_reverse=pi_r, state=state,
                    delta_xi=xi-old, phase_dissipation_endpoint=dissipation,
                    stress_residual=float(self.stress(e)-p),
                ))
        return rows


# 3. Adsorption-dependent material parameters
def cumulative_hazard(e):
    """L(e), Eq.(53) with a2=0; not the transformation strain H."""
    g = ETA1_MEV / (KB_MEV_PER_K * T_K)
    return A1 / g * np.expm1(g * np.asarray(e))


def hydrated_survival(e, N):
    n = N / V0_A3
    rate_factor = np.exp(-n * E_MU_MEV_A3 / (KB_MEV_PER_K * T_K))
    return np.exp(-rate_factor * cumulative_hazard(e))


def material_parameters_from_N(N=None, *, p_max=None, p_fs=None,
                               p_fe=None, p_rs=None, p_re=None,
                               K_cp=None, H=None, n_steps=None):
    """Return the calibrated-model Config and an auditable parameter report.

    Explicit arguments override top-of-file overrides. P_fs can be supplied
    for each N instead of using the survival criterion. Kcp and H are evaluated
    at that onset, then held fixed during this loading/unloading cycle.
    """
    N = float(N_VALUE if N is None else N)
    if not np.isfinite(N) or not 0 < N < N_REF_PER_A3 * V0_A3:
        raise ValueError("Require 0 < N < n_ref*V0. The confined-water Kcp law "
                         "does not determine the dry-material modulus at N=0.")
    p_fe = float(P_FE_MPA if p_fe is None else p_fe)
    p_rs = float(P_RS_MPA if p_rs is None else p_rs)
    p_re = float(P_RE_MPA if p_re is None else p_re)
    p_max = float(P_MAX_MPA if p_max is None else p_max)
    fs_input = P_FS_OVERRIDE_MPA if p_fs is None else p_fs
    cp_input = K_CP_OVERRIDE_MPA if K_cp is None else K_cp
    h_input = H_OVERRIDE if H is None else H
    n = N / V0_A3
    # Probe only the op-phase stress law. Its zero hardening is never used
    # for evolution; the final config always solves the five calibration equations.
    base = Config(
        K_op=K_OP_MPA, K_cp=K0_MPA, H=0.0, n=n,
        E0=E0_MEV, kE=K_E, k0=0.0, C=C, n_ref=N_REF_PER_A3,
        adsorption_scale=MEV_PER_A3_TO_MPA, use_Q=True,
        a1=A1, a2=0.0, eta1_meV=ETA1_MEV, eta2_meV=0.0, T_K=T_K,
        p_fs=0.0, p_fe=p_fe, p_rs=p_rs, p_re=p_re,
        delta=float(np.exp(-LOG_CUTOFF)),
        p_max=max(p_max, p_fe, p_rs, p_re, float(fs_input or 0)),
        n_steps=N_STEPS if n_steps is None else n_steps,
        reverse_enabled=False, calibrate_hardening=False,
        fixed_hardening=(0.0, 0.0, 0.0, 0.0, 0.0),
        pressure_label="Pressure, p (MPa)",
    )
    probe = Model(base)
    if fs_input is None:
        if not hydrated_survival(probe.branch_e_max, N) <= S_CR < 1:
            raise ValueError("Sm does not reach S_CR on the initial stable op branch.")
        e_fs = brentq(lambda e: float(hydrated_survival(e, N)-S_CR),
                      0.0, probe.branch_e_max, xtol=1e-14)
        p_fs = float(probe.stress(e_fs))
    else:
        p_fs = float(fs_input)
        e_fs = probe.solve_e(p_fs)
    if not 0 < p_fs < p_fe:
        raise ValueError(f"Need 0 < p_fs < p_fe, obtained {p_fs:g}, {p_fe:g} MPa.")
    hf = HF_DRY - n * BETA * V_W_A3
    e_fe = e_fs + hf
    if hf <= 0 or not 0 < e_fe < POROSITY:
        raise ValueError(f"Invalid completion geometry: hf={hf:g}, e_fe={e_fe:g}; "
                         "need hf>0 and 0<e_fe<porosity.")
    rho_water_si = WATER_MASS_KG * (N/V0_M3) / (POROSITY-e_fe)
    K_cp_law = (ALPHA_H * K0_MPA / TAIT_M * (rho_water_si/RHO0_KG_M3)**TAIT_M
                + (1.0-ALPHA_H/TAIT_M)*K0_MPA)
    K_cp = float(K_cp_law if cp_input is None else cp_input)
    if K_cp <= 0:
        raise ValueError("K_cp must be positive.")
    H_law = e_fe - p_fe/K_cp
    H = float(H_law if h_input is None else h_input)
    config = replace(base, p_fs=p_fs, K_cp=K_cp, H=H, p_max=p_max,
                     calibrate_hardening=True, fixed_hardening=None)
    report = dict(
        N_per_pore=N, n_per_A3=n, V0_per_pore_A3=V0_A3,
        cp_reference_parameters=dict(alpha_H=ALPHA_H, K0_atm=K0_ATM,
            K0_MPa=K0_MPA, m=TAIT_M, porosity=POROSITY, V0_m3=V0_M3,
            water_mass_kg=WATER_MASS_KG, rho0_kg_m3=RHO0_KG_M3),
        p_fs_source="survival criterion" if fs_input is None else "manual override",
        K_cp_source="Eq.(60)" if cp_input is None else "manual override",
        H_source="Eq.(63)-(64)" if h_input is None else "manual override",
        onset_absolute_strain=e_fs, hf=hf, completion_target_absolute_strain=e_fe,
        hf_parameters=dict(hf_dry=HF_DRY, beta=BETA, water_volume_A3=V_W_A3),
        completion_water_density_kg_m3=rho_water_si,
        completion_water_density_g_cm3=rho_water_si/1000.0, K_cp_law_MPa=K_cp_law,
        H_law=H_law, survival_at_onset=float(hydrated_survival(e_fs, N)),
        survival_threshold=S_CR,
        rate_factor=float(np.exp(-n*E_MU_MEV_A3/(KB_MEV_PER_K*T_K))),
        assumptions=[
            "Sm uses exp(-n*E_mu/kBT), the user-approved barrier-increase correction to Eq.(62).",
            "Bare-framework Q remains the energy weight; Sm determines onset only.",
            "Kop is the user value 12.26 GPa; pfe is the user value 757.5 MPa by default.",
            "Omega(n,0)=0; an absolute adsorption-energy offset is not specified.",
            "T=300 K is the continued fixed-temperature assumption.",
            "N is molecules per pore; V0 is volume per pore, not full unit-cell volume.",
            "prs=pre=0 closes calibration; HKUST-1 transformation remains irreversible.",
            "H uses the manuscript xi=1 completion limit; actual cutoff is xi=1-exp(-6).",
            "No extra strain cap of 0.058 is imposed; adsorption domain and stable branch are checked.",
        ],
    )
    return config, report


# 4. Numerical verification
def diagnostics(model, rows):
    load = [r for r in rows if r["leg"] == "loading"]
    unload = [r for r in rows if r["leg"] == "unloading"]
    active = [abs(r["pi_forward"]-model.hardening.Y) for r in rows if r["state"] == "forward"]
    active += [abs(r["pi_reverse"]+model.hardening.Y) for r in rows if r["state"] == "reverse"]
    eps = np.linspace(max(model.e_zero, 1e-4), model.solve_e(model.c.p_max), 80)
    de = 2e-7
    dW = (model.energy(eps+de)-model.energy(eps-de))/(2*de)
    err = np.max(np.abs(dW-model.stress(eps)))
    return dict(
        initial_absolute_strain=load[0]["strain_absolute"],
        initial_pressure=load[0]["pressure"],
        zero_pressure_equilibrium_strain=model.e_zero,
        peak_relative_strain=load[-1]["strain"],
        residual_relative_strain=unload[-1]["strain"],
        peak_xi=load[-1]["xi"], final_xi=unload[-1]["xi"],
        reverse_occurred=any(r["delta_xi"] < 0 for r in unload),
        first_positive_xi_sample=next((r["pressure"] for r in load if r["xi"] > 0), None),
        max_stress_residual=max(abs(r["stress_residual"]) for r in rows),
        max_active_transformation_residual=max(active, default=0.0),
        max_calibration_residual=(float(max(abs(model.calibration_residuals)))
                                  if model.calibration_residuals is not None else None),
        energy_derivative_stress_error=float(err),
        loading_xi_nondecreasing=bool(np.all(np.diff([r["xi"] for r in load]) >= -1e-12)),
        unloading_xi_nonincreasing=bool(np.all(np.diff([r["xi"] for r in unload]) <= 1e-12)),
        open_phase_branch_pressure_limit=model.branch_p_max,
        final_state=unload[-1]["state"],
        forward_events=model.forward_events(),
        hardening_recalibrated=model.c.calibrate_hardening,
        notes=[
            "T=300 K is a provisional fixed-temperature assumption unless edited.",
            "Pressures and moduli use MPa; adsorption energies are converted from meV/A^3.",
            "Omega(n,0)=0 at fixed n. Calibration absorbs this phase-energy reference choice.",
            "Q(e_op) is a state-dependent energy weight, not independent irreversible damage kinetics.",
            "Direction-dependent f is the prescribed phenomenological branch model; this is not a proof of a single global potential.",
            "At b2=0 and pi=-Y, xi is not determined uniquely. The configured equality rule supplies that choice.",
            "The forward completion convention is xi=1-delta; inactive upper/lower bounds need not satisfy active equality.",
        ],
    )




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
    N = report["N_per_pore"]
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
    c, report = material_parameters_from_N(
        args.n, p_max=args.p_max, p_fs=args.p_fs, p_fe=args.p_fe,
        p_rs=args.p_rs, p_re=args.p_re,
        K_cp=None if args.k_cp is None else args.k_cp*1000, H=args.h,
    )
    if args.inspect:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    model = Model(c)
    rows = model.simulate()
    checks = diagnostics(model, rows)
    events = model.forward_events()
    for event, pressure in (("onset", c.p_fs), ("completion", c.p_fe)):
        if pressure <= c.p_max:
            if events[event] is None or abs(events[event]-pressure) >= 1e-5:
                raise RuntimeError("Transformation event verification failed.")
        elif events[event] is not None:
            raise RuntimeError("Unexpected transformation event above maximum pressure.")
    if (checks["max_stress_residual"] >= 1e-5
            or checks["max_active_transformation_residual"] >= 1e-5
            or checks["max_calibration_residual"] >= 1e-6
            or not checks["loading_xi_nondecreasing"]
            or not checks["unloading_xi_nonincreasing"]
            or checks["reverse_occurred"]):
        raise RuntimeError(f"Cycle verification failed: {checks}")
    N = report["N_per_pore"]
    out = Path(__file__).resolve().parent / OUTPUT_DIR / f"N={N:g}"
    path = export_plot(model, rows, report, out, args.show or SHOW_FIGURE)
    print(f"N={N:g}; Kcp={c.K_cp/1000:.6f} GPa; saved {path}")


if __name__ == "__main__":
    main()
