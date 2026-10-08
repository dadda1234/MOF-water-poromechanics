"""Plot the MIL-53 adsorption-stress curve from prescribed parameters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, gammaincc


# Model parameters
E0 = 240.98 * 160.22  # 1 meV/Angstrom^3 = 160.22 MPa
kE = 0.625
C = 2.55
n_ref = 0.0315  # Angstrom^-3
V0 = 793.65

# Water loading N and its maximum evaluated strain
STRAIN_RANGES = {
    1: 0.142,
    2: 0.154,
    3: 0.168,
    4: 0.184,
    5: 0.200,
}

OUTPUT_FILE = Path(__file__).with_name("pa_vs_eps_mil53.png")
TXT_FILE = Path(__file__).with_name("pa_values_mil53.txt")


def model_pa(eps: np.ndarray, N: float) -> np.ndarray:
    """Return adsorption stress for a water loading and strain array."""
    n = N / V0
    denom = 1.0 - C * eps
    n0 = n_ref * denom
    a = np.log(n0 / n)

    if np.any(denom <= 0) or np.any(a <= 0):
        raise ValueError(f"The strain range for n={n:g} is outside the model domain.")

    i_sqrt = n0 * gamma(1.5) * gammaincc(1.5, a)
    i_inv_sqrt = n0 * gamma(0.5) * gammaincc(0.5, a)
    E_eps = E0 * np.exp(kE * eps)

    return -E_eps * kE * i_sqrt + E_eps * C * i_inv_sqrt / (2.0 * denom)


def strain_points(eps_max: float, step: float = 0.002) -> np.ndarray:
    """Create equally spaced samples while retaining the exact endpoint."""
    eps = np.arange(int(np.floor(eps_max / step)) + 1, dtype=float) * step
    if not np.isclose(eps[-1], eps_max):
        eps = np.append(eps, eps_max)
    return eps


def export_txt() -> None:
    """Export N, strain, and adsorption stress as tab-separated values."""
    rows = []
    for n, eps_max in STRAIN_RANGES.items():
        eps = strain_points(eps_max)
        pa = model_pa(eps, n)
        rows.extend(zip(np.full(eps.size, n), eps, pa))

    np.savetxt(
        TXT_FILE,
        np.asarray(rows),
        delimiter="\t",
        header="n\teps\tpa",
        comments="",
        fmt=["%.0f", "%.4f", "%.10g"],
    )
    print(f"Data saved to: {TXT_FILE}")


def main() -> None:
    export_txt()
    fig, ax = plt.subplots(figsize=(8, 6))

    for n, eps_max in STRAIN_RANGES.items():
        eps = np.linspace(0.0, eps_max, 300)
        ax.plot(eps, model_pa(eps, n), linewidth=2, label=f"n = {n}")

    ax.set_xlabel(r"Strain $\varepsilon$")
    ax.set_ylabel(r"Adsorption stress $p_a$")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300)
    print(f"Figure saved to: {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
