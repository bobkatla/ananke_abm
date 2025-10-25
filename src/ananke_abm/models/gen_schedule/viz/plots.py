import os
import numpy as np
import matplotlib.pyplot as plt

def plot_unaries_summary(U_mean_logits, U_std_logits, purposes, outdir):
    """
    U_mean_logits: (T, P) float32
    U_std_logits:  (T, P) float32
    purposes: list of length P with purpose names in index order
    outdir: directory to save plots
    """
    os.makedirs(outdir, exist_ok=True)

    T, P = U_mean_logits.shape
    time_axis = np.arange(T)  # these are bins; you can convert to minutes if you want

    for p in range(P):
        mean_curve = U_mean_logits[:, p]
        std_curve = U_std_logits[:, p]

        upper = mean_curve + std_curve
        lower = mean_curve - std_curve

        plt.figure()
        plt.fill_between(
            time_axis,
            lower,
            upper,
            alpha=0.2,
            linewidth=0,
        )
        plt.plot(time_axis, mean_curve, linewidth=2)
        plt.title(f"Decoder logits over time: {purposes[p]}")
        plt.xlabel("time bin")
        plt.ylabel("logit (mean ± 1 std)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"unaries_{p}_{purposes[p]}.png"))
        plt.close()

def plot_unaries_mean(U_mean, purposes, outdir):
    os.makedirs(outdir, exist_ok=True)
    L,P = U_mean.shape
    for p in range(P):
        plt.figure()
        plt.plot(U_mean[:,p])
        plt.title(f"Mean logits over time: {purposes[p]}")
        plt.xlabel("t")
        plt.ylabel("logit")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"unaries_{p}_{purposes[p]}.png"))
        plt.close()

def plot_minutes_share(share_syn, share_ref, purposes, outpath):
    idx = np.arange(len(purposes))
    width = 0.35
    plt.figure()
    plt.bar(idx - width/2, share_ref, width, label="ref")
    plt.bar(idx + width/2, share_syn, width, label="synth")
    plt.xticks(idx, purposes, rotation=45, ha="right")
    plt.ylabel("share (fraction)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath); plt.close()

def plot_tod_marginal(m_ref, m_syn, purposes, outdir):
    os.makedirs(outdir, exist_ok=True)
    L,P = m_ref.shape
    for p in range(P):
        plt.figure()
        plt.plot(m_ref[:,p], label="ref")
        plt.plot(m_syn[:,p], label="synth")
        plt.title(f"ToD marginal: {purposes[p]}")
        plt.xlabel("time bin")
        plt.ylabel("probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"tod_{p}_{purposes[p]}.png"))
        plt.close()

def plot_bigram_delta(B_ref, B_syn, purposes, outdir):
    os.makedirs(outdir, exist_ok=True)
    D = np.abs(B_ref - B_syn)
    plt.figure()
    plt.imshow(D, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(purposes)), purposes, rotation=45, ha="right")
    plt.yticks(range(len(purposes)), purposes)
    plt.title("|Bigram Δ| (ref vs synth)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bigram_delta.png"))
    plt.close()
