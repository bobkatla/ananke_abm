
import matplotlib.pyplot as plt
import os
def plot_unaries_mean(U_mean, purposes, outdir):
    os.makedirs(outdir, exist_ok=True)
    L,P = U_mean.shape
    for p in range(P):
        plt.figure()
        plt.plot(U_mean[:,p])
        plt.title(f"Mean logits over time: {purposes[p]}")
        plt.xlabel("t"); plt.ylabel("logit")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"unaries_{p}_{purposes[p]}.png"))
        plt.close()
