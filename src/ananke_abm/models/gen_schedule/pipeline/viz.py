import numpy as np
import os
import json
import click
from ananke_abm.models.gen_schedule.utils.cfg import ensure_dir
from ananke_abm.models.gen_schedule.evals.metrics import tod_marginals, bigram_matrix, minutes_share
from ananke_abm.models.gen_schedule.viz.plots import plot_unaries_mean, plot_minutes_share, plot_tod_marginal, plot_bigram_delta

def visualize(samples_npz_path, samples_meta_path, outdir_path, reference_grid_path):
    """
    Produce sanity plots for a sampled population:
    - Mean unaries over time (U_mean_logits)
    - Minutes share bars (synth vs ref)
    - Time-of-day marginals per purpose (synth vs ref)
    - Bigram delta heatmap
    No model or GPU required.
    """
    ensure_dir(outdir_path)
    # load generated population artifact
    synth_npz = np.load(samples_npz_path)
    generated_labels = synth_npz["Y_generated"].astype(np.int64)     # (N, L)
    U_mean_logits = synth_npz["U_mean_logits"].astype(np.float32)    # (L, P)

    with open(samples_meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)

    purpose_map = meta["purpose_map"]  # {purpose_name: index}
    purpose_names_ordered = meta["purpose_names_ordered"]  # [idx0_name, idx1_name, ...]
    P = len(purpose_names_ordered)

    # compute synth stats
    synth_minutes_share = minutes_share(generated_labels, P)
    synth_tod = tod_marginals(generated_labels, P)
    synth_bigram = bigram_matrix(generated_labels, P)

    # load reference stats if provided
    if reference_grid_path and os.path.exists(reference_grid_path):
        ref_npz = np.load(reference_grid_path)
        reference_labels = ref_npz["Y"].astype(np.int64)
        ref_minutes_share = minutes_share(reference_labels, P)
        ref_tod = tod_marginals(reference_labels, P)
        ref_bigram = bigram_matrix(reference_labels, P)
    else:
        # fallback: compare synth to itself just to make plots work
        ref_minutes_share = synth_minutes_share
        ref_tod = synth_tod
        ref_bigram = synth_bigram

    # 1. Mean unaries (logits over time)
    plot_unaries_mean(
        U_mean_logits,                    # (L,P)
        purpose_names_ordered,            # names aligned with P
        os.path.join(outdir_path, "unaries")
    )
    # 2. Minutes share bar chart
    plot_minutes_share(
        share_syn=synth_minutes_share,
        share_ref=ref_minutes_share,
        purposes=purpose_names_ordered,
        outpath=os.path.join(outdir_path, "minutes_share.png"),
    )
    # 3. Time-of-day marginal curves per purpose
    plot_tod_marginal(
        m_ref=ref_tod,
        m_syn=synth_tod,
        purposes=purpose_names_ordered,
        outdir=os.path.join(outdir_path, "tod"),
    )
    # 4. Bigram delta heatmap
    plot_bigram_delta(
        B_ref=ref_bigram,
        B_syn=synth_bigram,
        purposes=purpose_names_ordered,
        outdir=os.path.join(outdir_path, "bigrams"),
    )
    click.echo(f"Saved plots to {outdir_path}")

