import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from labellines import labelLine, labelLines


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
    rtol_min = np.sqrt(np.log(2)/(2*n))
    rtol = np.linspace(rtol_min, rtol_max, plot_samples)
    delta = hoeffding_bound(n, rtol)
    return (rtol, delta)


def plot_hoeffding_bound(n_max_power=9, ymin=1e-8):
    fig, ax = plt.subplots(figsize=(8, 4))
    sample_sizes = np.unique(np.rint(10**np.linspace(0, n_max_power, 10000)))
    lc = LineCollection(
        [np.column_stack(bound_frontier(n, ymin)) for n in sample_sizes],
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=10**n_max_power),
        linewidths=1.0,
    )

    lc.set_array(sample_sizes)
    ax.add_collection(lc)

    for k in range(n_max_power):
        ax.plot(
            *bound_frontier(10**k, ymin),
            linestyle="dotted",
            marker=None,
            color="black",
            linewidth=1.3 if k > 1 else 1.7,
            label=f"$10^{k}$"
        )
        # line= ax.get_lines()[-1]
        # labelLine(line, )

    labelLines(ax.get_lines(), zorder=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Axes
    ax.set_xlim(0.0001, 1)
    ax.set_ylim(ymin, 1.0)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.labelpad = 15
    ax.set_xlabel(r"$\frac{\epsilon}{b-a}$", fontsize=14)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(r"$\delta$", fontsize=14)

    ax.grid(True, which="both", alpha=0.3)

    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ax.set_xticks(np.arange(0, 1.1, step=0.1))

    cbar = fig.colorbar(lc, ax=ax, location="left", pad=0.02)
    cbar.set_label(r"$n$", fontsize=14)

    plt.tight_layout()
    fig.savefig("plots/hoeffding_bound.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_hoeffding_bound()