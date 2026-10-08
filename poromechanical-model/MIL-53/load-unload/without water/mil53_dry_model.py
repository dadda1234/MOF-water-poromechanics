"""Dry MIL-53: two-way survival, quadratic hardening, reversible full unloading.
Independent of the water-containing models. Kcp and H are direct inputs.
The bare open-phase stable branch is required only while open phase remains.
Run: python mil53_dry_model.py [--show]. Units: pressure MPa, input moduli GPa.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import mil53_dry_config as cfg


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
        return ((self.b1 if forward else self.b2) * np.asarray(xi)
                 + self.mu1 + (self.mu2 if forward else -self.mu2))

    def energy(self, xi, forward=True):
        return (0.5 * (self.b1 if forward else self.b2) * np.asarray(xi) ** 2
                 + (self.mu1 + (self.mu2 if forward else -self.mu2)) * np.asarray(xi))


# 3. Phase evolution and full-cycle response

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
        slopes = np.array([c.H + p / c.K_cp - op.solve_e(p) for p in ps])
        if min(slopes) <= 0:
            raise ValueError("I(p) is not strictly increasing in the transformation interval; "
                             "pressure-direction phase selection is not justified.")
        self.min_force_slope = float(min(slopes))
        A, B, C, D = [self.force(p) for p in (c.p_fs, c.p_fe, c.p_rs, c.p_re)]
        b1, b2 = B - A, C - D
        mu2 = (b2 - b1) / 4
        self.hardening = Hardening(b1, b2, (A + D) / 2, mu2, (A - D) / 2 - mu2)
        if b1 < 0 or b2 < 0 or self.hardening.Y < 0:
            raise ValueError("Calibration produced b1<0, b2<0 or Y<0; review critical pressures.")
        h = self.hardening
        self.calibration_residuals = [A - h.derivative(0) - h.Y, B - h.derivative(1) - h.Y,
            C - h.derivative(1, False) + h.Y, D - h.derivative(0, False) + h.Y,
            h.energy(1) - h.energy(1, False)]
        if max(abs(float(r)) for r in self.calibration_residuals) > 1e-7:
            raise RuntimeError("Five-equation calibration check failed.")

    def force(self, p, e=None):
        e = self.op.solve_e(p) if e is None else e
        return float(self.op.energy(e) - p * e + p * p / (2 * self.c.K_cp) + p * self.c.H)

    def update_xi(self, p, old, forward):
        c, h = self.c, self.hardening
        if forward:
            if old == 1:
                return old, "closed_pore"
            if p < c.p_fs - 1e-9:
                return old, "forward_stick"
            if h.b1 == 0:
                return 1.0, "forward_plateau_jump"
            if p >= c.p_fe - 1e-9:
                return 1.0, "forward_complete" if old < 1 else "closed_pore"
            target = (self.force(p) - h.mu1 - h.mu2 - h.Y) / h.b1
            xi = float(np.clip(max(old, target), 0, 1))
            return xi, "forward" if xi > old else "forward_stick"
        if old == 0:
            return 0.0, "open_pore"
        if p > c.p_rs + 1e-9:
            return old, "reverse_stick"
        if h.b2 == 0:
            return 0.0, "reverse_plateau_jump"
        if p <= c.p_re + 1e-9:
            return 0.0, "reverse_complete"
        target = (self.force(p) - h.mu1 + h.mu2 + h.Y) / h.b2
        xi = float(np.clip(min(old, target), 0, 1))
        return xi, "reverse" if xi < old else "reverse_stick"

    def row(self, p, xi, old, forward, state):
        e = self.op.solve_e(p) if p <= self.op.p_max else None
        if e is None and xi != 1:
            raise ValueError("An open-pore fraction remains but its stable strain cannot be inverted.")
        strain = xi * (p / self.c.K_cp + self.c.H) + (1 - xi) * (0 if e is None else e)
        q = float(self.op.q(e)) if e is not None else None
        force = self.force(p, e) if e is not None else None
        h = self.hardening
        pi_f = force - float(h.derivative(xi)) if force is not None else None
        pi_r = force - float(h.derivative(xi, False)) if force is not None else None
        active = pi_f if forward else pi_r
        dx = xi - old
        dissipation = 0.0 if dx == 0 else active * dx
        if dissipation < -1e-7:
            raise ValueError("Negative phase-transformation dissipation.")
        return dict(leg="loading" if forward else "unloading", pressure=p,
            epsilon=strain, strain_absolute=strain, strain_relative=strain - self.op.e_zero,
            xi=xi, epsilon_t=self.c.H * xi, e_op=e, e_cp=p / self.c.K_cp,
            Q=q, W_op=None if e is None else float(self.op.energy(e)),
            mechanical_force=force, pi_forward=pi_f, pi_reverse=pi_r, state=state,
            delta_xi=dx, phase_dissipation_endpoint=dissipation,
            op_stress_residual=None if e is None else float(self.op.stress(e) - p))

    def simulate(self):
        c = self.c
        events = [p for p in (c.p_fs, c.p_fe, c.p_rs, c.p_re) if p <= c.p_max]
        load_p = np.unique(np.r_[np.linspace(0, c.p_max, c.n_steps), events])
        if self.op.p_min < 0:
            count = max(3, int(np.ceil(-self.op.p_min / c.p_max * (c.n_steps - 1))) + 1)
            load_p = np.r_[np.linspace(self.op.p_min, 0, count)[: -1], load_p]
        rows, xi = [], 0.0
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
            fn = lambda p: self.force(p) - float(self.hardening.derivative(xi, False)) + self.hardening.Y
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
                                   [] if reverse_onset is None else [reverse_onset]])[: : -1]
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
    fe = fs if fe is None else fe
    re = rs if re is None else re
    report = dict(N=0, K_op_GPa=cfg.K_OP_GPA, K_cp_GPa=cfg.K_CP_GPA, H=cfg.H,
                critical_strain=e_fs, critical_probability=float(op.q(e_fs)),
                p_fs=fs, p_fe=fe, p_rs=rs, p_re=re, p_max=maximum,
                stable_branch_limit_MPa=op.p_max)
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
