"""Dry HKUST-1: fixed open-phase reference beyond the onset pressure.

Before onset, use the original stable survival-weighted elastic branch.
Above onset, hold e_op=e_fs and W_op=W_fs, rather than invert that branch.
Equivalently, continue g_op(p)=g_op(pfs)-e_fs*(p-pfs) linearly.
This is an added phenomenological closure, NOT the original softening law
and NOT a claim that the failed open material can carry no stress.
Retain logarithmic hardening, xi<=1-exp(-6), and irreversible unloading.
Run: python hkust1_dry_model.py [--show]. Pressure MPa, input moduli GPa.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import hkust1_dry_config as cfg


# 1. Open-pore elastic response

class OpenPore:
    """Dry elastic energy W=Kop*Q(e)*e^2/2 on the initial stable branch."""
    def __init__(self):
        self.K_op = cfg.K_OP_GPA * 1000.0
        self.g1 = cfg.ETA1_MEV / (cfg.KB_MEV_PER_K * cfg.T_K)
        self.g2 = cfg.ETA2_MEV / (cfg.KB_MEV_PER_K * cfg.T_K)
        if min(self.K_op, cfg.T_K, cfg.DRY_STRAIN_LIMIT) <= 0:
            raise ValueError("Kop, temperature and strain limit must be positive.")
        if min(cfg.A1, cfg.A2, cfg.ETA1_MEV, cfg.ETA2_MEV) < 0:
            raise ValueError("Survival parameters must be nonnegative.")
        self.solution = solve_ivp(
            lambda e, q: self.q_prime(e, q), (0.0, cfg.DRY_STRAIN_LIMIT), [1.0],
            method="DOP853", dense_output=True, rtol=cfg.ODE_RTOL,
            atol=cfg.ODE_ATOL, max_step=cfg.DRY_STRAIN_LIMIT / 500,
        )
        if not self.solution.success:
            raise RuntimeError(self.solution.message)
        es = np.linspace(0.0, cfg.DRY_STRAIN_LIMIT, 16385)
        qs = self.q(es)
        if qs.min() < -1e-9 or qs.max() > 1 + 1e-9:
            raise ValueError("Survival probability left [0,1].")
        ps = self.stress(es)
        turning = np.flatnonzero(np.diff(ps) <= 0)
        self.branch_e_max = float(es[turning[0]]) if len(turning) else cfg.DRY_STRAIN_LIMIT
        self.p_min = 0.0
        self.p_max = float(self.stress(self.branch_e_max))
        self.e_zero = 0.0

    def q_prime(self, e, q):
        e = np.asarray(e)
        return -cfg.A1 * np.exp(self.g1 * e) * q + cfg.A2 * np.exp(-self.g2 * e) * (1 - q)

    def q(self, e):
        if np.any(np.asarray(e) < 0) or np.any(np.asarray(e) > cfg.DRY_STRAIN_LIMIT):
            raise ValueError("Strain outside the integrated domain.")
        return self.solution.sol(e)[0]

    def energy(self, e):
        return 0.5 * self.K_op * self.q(e) * np.asarray(e) ** 2

    def stress(self, e):
        e = np.asarray(e)
        q = self.q(e)
        return self.K_op * (q * e + 0.5 * self.q_prime(e, q) * e ** 2)

    def solve_e(self, p):
        if not 0.0 <= p <= self.p_max:
            raise ValueError(f"Pressure {p:g} MPa is outside the initial stable dry branch [0,{self.p_max:g}] MPa.")
        if p == 0:
            return 0.0
        return brentq(lambda e: float(self.stress(e) - p), 0.0, self.branch_e_max, xtol=2e-14, rtol=2e-14)

    def critical_strain(self):
        if cfg.EPS_CR_DRY is not None:
            e = float(cfg.EPS_CR_DRY)
        else:
            if not 0 < cfg.S_CR < 1:
                raise ValueError("Require 0<S_CR<1.")
            if self.q(self.branch_e_max) > cfg.S_CR:
                raise ValueError("The survival threshold is not reached on the stable branch.")
            e = brentq(lambda e: float(self.q(e) - cfg.S_CR), 0.0, self.branch_e_max, xtol=1e-14)
        if not 0 < e < self.branch_e_max:
            raise ValueError("Critical strain must lie on the initial stable branch.")
        return e


# 2. Cycle configuration and branch hardening

@dataclass(frozen=True)
class CycleConfig:
    K_cp: float
    H: float
    p_fs: float
    p_fe: float
    p_rs: float
    p_re: float
    p_max: float
    n_steps: int
    delta: float = float(np.exp(-cfg.LOG_CUTOFF))
    reverse_enabled: bool = False
    calibrate_hardening: bool = True
    fixed_hardening: tuple | None = None
    degenerate_reverse_at_equality: str = "hold"

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


# 3. Phase evolution and full-cycle response

class Model:
    def __init__(self, op, c):
        self.op, self.c = op, c
        if min(c.K_cp, c.p_max) <= 0 or c.H < 0 or c.n_steps < 3:
            raise ValueError("Require Kcp,p_max>0, H>=0 and n_steps>=3.")
        if not 0 <= c.p_re <= c.p_rs <= c.p_fs < c.p_fe:
            raise ValueError("Require 0<=pre<=prs<=pfs<pfe.")
        if not 0 < c.delta < 0.01:
            raise ValueError("Require 0<delta<.01.")
        self.xi_max = 1 - c.delta
        self.e_zero = 0.0
        self.branch_p_min = 0.0
        self.branch_p_max = op.p_max
        self.branch_e_max = op.branch_e_max
        self.e_fs = op.solve_e(c.p_fs)
        self.W_fs = float(op.energy(self.e_fs))
        if cfg.POST_ONSET_RULE != "fixed_open_reference":
            raise ValueError("Supported onset closure: fixed_open_reference.")
        # A frozen strain reference gives zero open-phase compliance above pfs.
        # It is an effective continuation, not a stress-free failed phase.
        probes = np.linspace(c.p_re, c.p_fe, 301)
        slopes = [c.H + p / c.K_cp - self.open_strain(p) for p in probes]
        if min(slopes) <= 0:
            raise ValueError("The effective transformation force is not monotone.")
        self.hardening, self.calibration_residuals = self.calibrate()

    def open_strain(self, p):
        """Invert the pre-onset branch; use a fixed reference above onset.

        On unloading below pfs, the small residual open fraction follows the
        original elastic branch. The transformed fraction remains irreversible.
        """
        if p <= self.c.p_fs:
            return self.op.solve_e(p)
        return self.e_fs

    def stress(self, e):
        return self.op.stress(e)

    def energy(self, e):
        return self.op.energy(e)

    def mechanical_force(self, p, e=None):
        e = self.open_strain(p) if e is None else e
        return float(self.energy(e) - p * e + p * p / (2 * self.c.K_cp) + p * self.c.H)

    def calibrate(self):
        c = self.c
        ld = np.log(c.delta)
        # Unknown vector: b1,b2,mu1,mu2,Y. Fifth row is f+(1)=f-(1).
        mat = np.array([
            [-2.0, 0.0, 1.0, 1.0, 1.0],
            [-ld - 2.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, ld + 1.0, 1.0, -1.0, -1.0],
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
        values[: 2] = np.maximum(values[: 2], 0.0)
        values[: 2][values[: 2] < 1e-12] = 0.0
        return Hardening(* map(float, values)), (mat @ values - rhs if c.calibrate_hardening else None)

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
        # Include exact onset and completion pressures in the sampling grid.
        events = self.forward_events()
        p_values = np.unique(np.r_[np.linspace(0.0, c.p_max, c.n_steps),
                                  [p for p in events.values() if p is not None],
                                  [p for p in (c.p_rs, c.p_re) if 0 <= p <= c.p_max]])
        # Extend only loading to zero strain. Preserve the original p>=0 grid
        # and the zero-pressure unloading endpoint (no tensile unloading cycle).
        if self.branch_p_min < 0:
            negative_steps = max(3, int(np.ceil(-self.branch_p_min / c.p_max * (c.n_steps - 1))) + 1)
            negative_p = np.linspace(self.branch_p_min, 0.0, negative_steps)[: -1]
            p_values = np.r_[negative_p, p_values]
        zero_index = int(np.searchsorted(p_values, 0.0))
        e_values = np.array([self.open_strain(float(p)) for p in p_values])
        forces = np.array([self.mechanical_force(p, e) for p, e in zip(p_values, e_values)])
        if np.any(np.diff(forces) < -1e-7):
            raise ValueError("Mechanical driving force is not monotone on this branch; "
                             "pressure-direction branch selection is not justified.")
        if forces[0] - self.hardening.forward_derivative(0.0) > self.hardening.Y + 1e-7:
            raise ValueError("The specified xi=0 initial state is unstable at the loading start.")
        rows = []
        xi = 0.0
        for loading, indices in ((True, range(len(p_values))),
                                 (False, range(len(p_values) - 1, zero_index - 1, -1))):
            for i in indices:
                p, e, force = float(p_values[i]), float(e_values[i]), float(forces[i])
                old = xi
                xi, state = self.update_xi(force, old, loading)
                q = float(self.op.q(e))
                strain = (1.0 - xi) * e + xi * p / c.K_cp + c.H * xi
                pi_f = force - float(self.hardening.forward_derivative(xi))
                pi_r = (force - float(self.hardening.reverse_derivative(xi))
                        if xi > 0 or self.hardening.b2 == 0 else None)
                pi_active = pi_f if loading else pi_r
                dissipation = 0.0 if xi == old else float(pi_active * (xi - old)) if pi_active is not None else None
                rows.append(dict(
                    leg="loading" if loading else "unloading", pressure=p,
                    strain=strain - self.e_zero, strain_absolute=strain, xi=xi,
                    epsilon_t=c.H * xi, e_op=e, e_cp=p / c.K_cp, Q=q,
                    W_op=float(self.energy(e)), mechanical_force=force,
                    pi_forward=pi_f, pi_reverse=pi_r, state=state,
                    delta_xi=xi - old, phase_dissipation_endpoint=dissipation,
                    response_mode="survival_elastic" if p <= c.p_fs else "fixed_open_reference",
                    op_stress_residual=float(self.stress(e) - p) if p <= c.p_fs else None,
                    reference_strain_residual=abs(e - self.e_fs) if p > c.p_fs else 0.0,
                ))
        return rows


# 4. Direct material parameters

def material_parameters(*, p_fs=None, p_fe=None, p_rs=None, p_re=None, p_max=None):
    op = OpenPore()
    fs_input = cfg.P_FS_OVERRIDE_MPA if p_fs is None else p_fs
    e_fs = op.critical_strain() if fs_input is None else op.solve_e(fs_input)
    fs = float(op.stress(e_fs)) if fs_input is None else float(fs_input)
    fe = cfg.P_FE_MPA if p_fe is None else p_fe
    rs = cfg.P_RS_MPA if p_rs is None else p_rs
    re = cfg.P_RE_MPA if p_re is None else p_re
    maximum = cfg.P_MAX_MPA if p_max is None else p_max
    # Missing completion and maximum pressures are reported, not invented.
    re = rs if re is None else re
    report = dict(N=0, K_op_GPa=cfg.K_OP_GPA, K_cp_GPa=cfg.K_CP_GPA, H=cfg.H,
                critical_strain=e_fs, critical_probability=float(op.q(e_fs)),
                p_fs=fs, p_fe=fe, p_rs=rs, p_re=re, p_max=maximum,
                stable_branch_limit_MPa=op.p_max, post_onset_rule=cfg.POST_ONSET_RULE,
                upper_phase_fraction=1 - float(np.exp(-cfg.LOG_CUTOFF)))
    missing = [name for name, value in (("P_FE_MPA", fe), ("P_MAX_MPA", maximum)) if value is None]
    report["missing_inputs"] = missing
    if missing:
        return op, None, report
    c = CycleConfig(K_cp=cfg.K_CP_GPA * 1000, H=cfg.H, p_fs=fs, p_fe=float(fe),
                  p_rs=float(rs), p_re=float(re), p_max=float(maximum), n_steps=cfg.N_STEPS)
    return op, c, report


# 5. Numerical verification

def diagnostics(model, rows):
    load = [r for r in rows if r["leg"] == "loading"]
    unload = [r for r in rows if r["leg"] == "unloading"]
    active = []
    for r in rows:
        if r["delta_xi"] > 0:
            active.append(abs(r["pi_forward"] - model.hardening.Y))
        elif r["delta_xi"] < 0:
            active.append(abs(r["pi_reverse"] + model.hardening.Y))
    result = dict(
        max_calibration_residual=float(np.max(np.abs(model.calibration_residuals))),
        max_stress_residual=max(abs(r["op_stress_residual"]) for r in rows if r["op_stress_residual"] is not None),
        max_active_residual=max(active, default=0.0),
        max_reference_strain_residual=max(r["reference_strain_residual"] for r in rows),
        loading_xi_nondecreasing=bool(np.all(np.diff([r["xi"] for r in load]) >= -1e-12)),
        unloading_xi_nonincreasing=bool(np.all(np.diff([r["xi"] for r in unload]) <= 1e-12)),
        minimum_phase_dissipation=min(r["phase_dissipation_endpoint"] for r in rows),
        loading_end_strain=load[-1]["strain_absolute"],
        unloading_zero_strain=unload[-1]["strain_absolute"],
        peak_xi=load[-1]["xi"], final_xi=unload[-1]["xi"])
    if (result["max_calibration_residual"] > 1e-7 or result["max_stress_residual"] > 1e-5
            or result["max_active_residual"] > 1e-5 or result["minimum_phase_dissipation"] < -1e-7
            or not result["loading_xi_nondecreasing"] or not result["unloading_xi_nonincreasing"]):
        raise RuntimeError(f"Cycle verification failed: {result}")
    return result


# 6. Figure export

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
        xlim=(0.0, 1.025 * model.c.p_max),
        ylim=(0.0, max(0.02, max(r["strain_absolute"] for r in rows) * 1.10)),
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


# 7. Command-line entry point

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p-fs", "p-fe", "p-rs", "p-re", "p-max"):
        parser.add_argument("--" + name, type=float, help="Pressure in MPa")
    parser.add_argument("--inspect", action="store_true", help="Print parameters; do not plot")
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window")
    args = parser.parse_args(argv)
    op, c, report = material_parameters(p_fs=args.p_fs, p_fe=args.p_fe, p_rs=args.p_rs,
                                    p_re=args.p_re, p_max=args.p_max)
    if args.inspect:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if c is None:
        parser.error("Provide dry-model inputs: " + ", ".join(report["missing_inputs"]) +
                     f". Stable open-phase pressure limit: {op.p_max:.6f} MPa.")
    try:
        model = Model(op, c)
        rows = model.simulate()
        checks = diagnostics(model, rows)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(__file__).resolve().parent / cfg.OUTPUT_DIR / "N=0"
    path = export_plot(model, rows, report, out, args.show or cfg.SHOW_FIGURE)
    print(json.dumps(dict(parameters=report, checks=checks, figure=str(path)), indent=2))

if __name__ == "__main__":
    main()
