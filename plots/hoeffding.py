import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "mathtext.fontset": "cm",
    'text.usetex': True,
})


def hoeffding_bound(n, rtol):
    """ Hoeffdings bound for number of samples and relative error tolerance
    P(|X̄ - μ| ≥ ε) ≤ 2 exp(-2 n ε^2 / (b-a)^2)
        where ε = rtol * (b-a)
    """
    return 2 * np.exp(-2 * n * rtol**2)


def bound_frontier(n, ymin, plot_samples=100):
    rtol_max = min(1.0, np.sqrt(np.log(2 / ymin) / (2 * n)))
    rtol = np.linspace(0, rtol_max, plot_samples)
    delta = hoeffding_bound(n, rtol)
    return np.column_stack((rtol, delta))


def plot_hoeffding_bound(n_max=int(1e3), ymin=1e-8):
    fig, ax = plt.subplots(figsize=(8, 4))

    sample_sizes = range(1, n_max + 1)
    lc = LineCollection(
        [bound_frontier(n, ymin) for n in sample_sizes],
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=n_max),
        linewidths=1.0,
    )

    lc.set_array(sample_sizes)
    ax.add_collection(lc)

    # Axes
    ax.set_xlim(0, 1)
    ax.set_ylim(ymin, 1.0)
    ax.set_yscale("log")

    ax.set_xlabel(r"$\frac{\epsilon}{b-a}$", fontsize=14)
    ax.set_ylabel(r"$\delta$", fontsize=14)

    ax.grid(True, which="both", alpha=0.3)

    ax.set_xticks(np.arange(0, 1.1, step=0.1))

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label(r"$n$", fontsize=14)

    plt.tight_layout()
    fig.savefig("plots/hoeffding_bound.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_hoeffding_bound()