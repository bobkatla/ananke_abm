import os
import json
import math
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ananke_abm.models.gen_schedule.pipeline.eval import make_report

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

@click.command("compare-samples")
@click.option("--ref_npz", type=click.Path(exists=True), required=True,
              help="Reference grid npz (e.g., train/val/export with Y).")
@click.option("--sample_dir", type=click.Path(exists=True), required=True,
              help="Directory containing sample .npz files (from different models).")
@click.option("--purpose_map", type=click.Path(exists=True), required=True,
              help="JSON file with purpose_map to include purpose names in reports.")
@click.option("--outdir", type=click.Path(), required=True,
              help="Directory to write reports and plots.")
def compare_samples(ref_npz, sample_dir, purpose_map, outdir):
    """
    Compare multiple generated sample sets against the same reference set.
    Saves per-model JSON, an aggregate CSV, and comparison plots.

    Notes:
      - All sample .npz must have the same (N, T).
      - Reference may have different N, but must have the same T.
    """
    _ensure_dir(outdir)

    # ---------- Load reference ----------
    ref = np.load(ref_npz)
    if "Y" not in ref:
        raise ValueError(f"{ref_npz} must contain 'Y' (reference label grid).")
    Y_ref = ref["Y"].astype(np.int64)       # (N_ref, T_ref)
    _, T_ref = Y_ref.shape

    # Optional reference ToD marginals (precomputed alongside ref_npz)
    ref_tod_path = ref_npz.replace(".npz", "_tod.npy")
    ref_tod = np.load(ref_tod_path) if os.path.exists(ref_tod_path) else None
    # load purpose_map for reports
    with open(purpose_map, "r", encoding="utf-8") as f:
        purpose_map_dict = json.load(f)

    # ---------- Find sample npz files ----------
    npz_files = sorted([os.path.join(sample_dir, f)
                        for f in os.listdir(sample_dir)
                        if f.endswith(".npz")])
    if not npz_files:
        raise ValueError(f"No .npz files found in {sample_dir}")

    # ---------- Load all samples & basic assertions ----------
    sample_infos = []
    for spath in npz_files:
        arr = np.load(spath)
        key = "Y_generated" if "Y_generated" in arr else ("Y" if "Y" in arr else None)
        if key is None:
            raise ValueError(f"{spath} must contain 'Y_generated' or 'Y'.")
        Y_syn = arr[key].astype(np.int64)  # (N_syn, T_syn)
        N_syn, T_syn = Y_syn.shape
        if T_syn != T_ref:
            raise AssertionError(
                f"Time bins mismatch: {spath} has T={T_syn}, but ref has T={T_ref}"
            )
        sample_infos.append({"path": spath, "name": _stem(spath), "Y": Y_syn, "N": N_syn})

    sample_infos.sort(key=lambda x: x["name"])

    # Enforce same number of records across all samples (fairness).
    Ns = [si["N"] for si in sample_infos]
    if len(set(Ns)) != 1:
        raise AssertionError(f"All samples must have same number of records; got {Ns}")

    # ---------- Evaluate each sample via existing make_report ----------
    reports = {}
    agg_rows = []
    for si in sample_infos:
        rpt = make_report(
            Y_synth=si["Y"],
            Y_ref=Y_ref,          # N_ref may differ; distributional metrics are fine
            purpose_map=purpose_map_dict,     # optional; metrics do not require names
            ref_tod=ref_tod
        )
        reports[si["name"]] = rpt

        # write per-model JSON
        with open(os.path.join(outdir, f"{si['name']}_report.json"), "w", encoding="utf-8") as f:
            json.dump(rpt, f, indent=2)

        # compact row for CSV
        ms_abs = rpt["minutes_share"]["abs_error"]
        row = {
            "model": si["name"],
            "bigram_L1": rpt["bigram"]["L1"],
            "tod_jsd_macro": rpt["tod_jsd_macro"],
            "all_home_rate": rpt.get("all_home_rate", math.nan),
            "start_home_rate": rpt.get("start_home_rate", math.nan),
            "end_home_rate": rpt.get("end_home_rate", math.nan),
            "diversity_ratio": rpt.get("diversity_ratio", math.nan),
            "minutes_share_abs_error_mean": float(np.mean(ms_abs)),
            "minutes_share_abs_error_max": float(np.max(ms_abs)),
        }
        for pidx, val in enumerate(ms_abs):
            row[f"ms_abs_p{pidx}"] = val
        agg_rows.append(row)

    # ---------- Save aggregate CSV ----------
    agg_df = pd.DataFrame(agg_rows).sort_values("model").reset_index(drop=True)
    csv_path = os.path.join(outdir, "comparison_summary.csv")
    agg_df.to_csv(csv_path, index=False)

    # ---------- Plots ----------
    models_sorted = list(agg_df["model"].values)

    # 1) Bigram L1
    bigram_vals = [reports[m]["bigram"]["L1"] for m in models_sorted]
    plt.figure(figsize=(10, 4))
    plt.bar(models_sorted, bigram_vals)
    plt.ylabel("Bigram L1 ↓")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cmp_bigram_L1.png"), dpi=150)
    plt.close()

    # 2) ToD JSD (macro)
    jsd_vals = [reports[m]["tod_jsd_macro"] for m in models_sorted]
    plt.figure(figsize=(10, 4))
    plt.bar(models_sorted, jsd_vals)
    plt.ylabel("ToD JSD (macro) ↓")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cmp_tod_jsd_macro.png"), dpi=150)
    plt.close()

    # 3) All-home rate
    ah_vals = [reports[m].get("all_home_rate", math.nan) for m in models_sorted]
    plt.figure(figsize=(10, 4))
    plt.bar(models_sorted, ah_vals)
    plt.ylabel("All-home rate ↓")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cmp_all_home_rate.png"), dpi=150)
    plt.close()

    # 4) Minutes-share absolute error per purpose, grouped by model
    any_rpt = next(iter(reports.values()))
    ms_ref = any_rpt["minutes_share"]["ref"]
    P = len(ms_ref)
    purpose_labels = [f"p{p}" for p in range(P)]

    width = 0.8 / max(1, len(models_sorted))
    x = np.arange(P)
    plt.figure(figsize=(max(10, P * 1.1), 5))
    for i, m in enumerate(models_sorted):
        ms_abs = reports[m]["minutes_share"]["abs_error"]
        plt.bar(x + i * width, ms_abs, width=width, label=m)
    plt.xticks(x + (len(models_sorted) - 1) * width / 2, purpose_labels, rotation=0)
    plt.ylabel("Minutes-share abs error ↓")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cmp_minutes_share_abs_error.png"), dpi=150)
    plt.close()

    # 5) Reference vs model minutes-share (side-by-side per purpose)
    plt.figure(figsize=(max(10, P * 1.1), 5))
    ref_share = np.array(ms_ref, dtype=float)
    bar_group_width = 0.12
    bar_positions = np.arange(P) * (1.0 + (len(models_sorted) + 1) * bar_group_width)
    plt.bar(bar_positions, ref_share, width=bar_group_width, label="ref")
    for i, m in enumerate(models_sorted):
        syn_share = np.array(reports[m]["minutes_share"]["synth"], dtype=float)
        plt.bar(bar_positions + (i + 1) * bar_group_width, syn_share,
                width=bar_group_width, label=m)
    plt.xticks(bar_positions + (len(models_sorted)) * bar_group_width / 2, purpose_labels)
    plt.ylabel("Minutes-share")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cmp_minutes_share_ref_vs_models.png"), dpi=150)
    plt.close()

    # 6) Macro table as PNG (handle mixed dtypes: model is str, metrics are numeric)
    display_cols = [
        "bigram_L1",
        "tod_jsd_macro",
        "all_home_rate",
        "start_home_rate",
        "end_home_rate",
        "diversity_ratio",
        "minutes_share_abs_error_mean",
        "minutes_share_abs_error_max",
    ]
    tbl = agg_df[["model"] + display_cols]
    # Build cell text with rounding for numeric cols only
    cell_text = []
    for _, row in tbl.iterrows():
        row_vals = [row["model"]] + [None] * len(display_cols)
        for j, col in enumerate(display_cols, start=1):
            val = row[col]
            try:
                row_vals[j] = f"{float(val):.4f}"
            except Exception:
                row_vals[j] = str(val)
        cell_text.append(row_vals)

    fig_h = 0.01 + 0.45 * max(1, len(models_sorted))
    fig_w = min(12, 3 + 0.25 * len(models_sorted))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=tbl.columns.tolist(), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_summary_table.png"), dpi=150)
    plt.close()

    # Save all per-model reports together as JSON
    with open(os.path.join(outdir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"Wrote per-model reports, comparison_summary.csv, and plots to {outdir}")
