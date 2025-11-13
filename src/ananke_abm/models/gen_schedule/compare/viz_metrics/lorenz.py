# lorenz.py
import numpy as np
import matplotlib.pyplot as plt


def lorenz_curve_from_counts(counts: np.ndarray):
    """
    counts: 1D non-negative array (e.g. frequencies per schedule or per activity).

    Returns:
        x: (K+1,) cumulative share of categories (from 0 to 1)
        y: (K+1,) cumulative share of mass (from 0 to 1)
        gini: scalar Gini coefficient in [0,1].
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1D, got shape {counts.shape}")

    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    total = counts.sum()
    if total <= 0:
        # degenerate: all zero, Lorenz is diagonal, Gini = 0
        x = np.linspace(0.0, 1.0, len(counts) + 1)
        y = x.copy()
        return x, y, 0.0

    # sort from smallest to largest
    sorted_counts = np.sort(counts)
    # cumulative share of mass
    cum_mass = np.cumsum(sorted_counts) / total  # (K,)
    # x-axis: cumulative share of categories
    K = len(counts)
    x = np.linspace(0.0, 1.0, K + 1)
    y = np.concatenate([[0.0], cum_mass])  # prepend 0

    # Gini = 1 - 2 * area_under_lorenz
    # approximate area via trapezoidal rule
    area = np.trapezoid(y, x)
    gini = 1.0 - 2.0 * area

    return x, y, float(gini)


def plot_lorenz_for_models(model_counts: dict, title: str = "Lorenz curves"):
    """
    model_counts: dict[str -> 1D array-like]
        E.g. {"ref": counts_ref, "cnn_crf": counts_syn, ...}

    Plots Lorenz curve for each model on the same figure,
    plus the equality line.
    Prints Gini for each model.
    """
    plt.figure(figsize=(6, 6))

    for name, counts in model_counts.items():
        x, y, gini = lorenz_curve_from_counts(np.asarray(counts, dtype=np.float64))
        print(f"{name}: Gini = {gini:.4f}")
        plt.plot(x, y, label=f"{name} (G={gini:.3f})")

    # equality line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="equality")

    plt.xlabel("Cumulative share of cells (sorted)")
    plt.ylabel("Cumulative share of probability/mass")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with fake data
    # Imagine these are schedule counts for ref and a collapsed model
    ref_counts = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    collapsed_counts = np.array([550, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    model_counts = {
        "ref": ref_counts,
        "collapsed": collapsed_counts,
    }

    plot_lorenz_for_models(model_counts, title="Example Lorenz curves")
