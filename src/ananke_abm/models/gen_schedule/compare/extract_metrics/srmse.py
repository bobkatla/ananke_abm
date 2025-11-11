# srmse.py

from typing import Dict, List
import os
import csv

from ananke_abm.models.gen_schedule.compare.utils import (
    ensure_dir,
    ngram_counts,
    schedule_counts,
    compute_srmse_from_counts,
)


def _metric_srmse_ngram_level(
    level_name: str,
    n: int,
    ref: Dict,
    models: List[Dict],
    outdir: str,
    as_schedule: bool = False,
):
    """
    Generic SRMSE computation for either:
      - full schedules (as_schedule=True, ignores n), or
      - n-grams of length n (as_schedule=False).

    Writes:
        srmse_<level_name>.csv with columns:
            model, srmse
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]  # (N_ref, T)

    # counts for reference
    if as_schedule:
        counts_ref = schedule_counts(Y_ref)
    else:
        counts_ref = ngram_counts(Y_ref, n=n, as_schedule=False)

    rows = []

    # reference row: SRMSE=0 by definition (self vs self)
    rows.append({"model": "ref", "srmse": 0.0})

    # models
    for m in models:
        Y_model = m["Y"]
        name = m["name"]
        if as_schedule:
            counts_syn = schedule_counts(Y_model)
        else:
            counts_syn = ngram_counts(Y_model, n=n, as_schedule=False)

        srmse_val = compute_srmse_from_counts(counts_ref, counts_syn)
        rows.append({"model": name, "srmse": float(srmse_val)})

    out_path = os.path.join(outdir, f"srmse_{level_name}.csv")
    fieldnames = ["model", "srmse"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def metric_srmse_schedule(ref: Dict, models: List[Dict], outdir: str):
    """
    SRMSE over full schedules (each entire daily sequence is a 'cell').
    """
    _metric_srmse_ngram_level(
        level_name="schedule",
        n=0,
        ref=ref,
        models=models,
        outdir=outdir,
        as_schedule=True,
    )


def metric_srmse_bigram(ref: Dict, models: List[Dict], outdir: str):
    """
    SRMSE over bigrams (n=2).
    """
    _metric_srmse_ngram_level(
        level_name="bigram",
        n=2,
        ref=ref,
        models=models,
        outdir=outdir,
        as_schedule=False,
    )


def metric_srmse_trigram(ref: Dict, models: List[Dict], outdir: str):
    """
    SRMSE over trigrams (n=3).
    """
    _metric_srmse_ngram_level(
        level_name="trigram",
        n=3,
        ref=ref,
        models=models,
        outdir=outdir,
        as_schedule=False,
    )


def metric_srmse_quadgram(ref: Dict, models: List[Dict], outdir: str):
    """
    SRMSE over quadgrams (n=4).
    """
    _metric_srmse_ngram_level(
        level_name="quadgram",
        n=4,
        ref=ref,
        models=models,
        outdir=outdir,
        as_schedule=False,
    )


SRMSE_FUNCS = {
    "srmse_schedule": metric_srmse_schedule,
    "srmse_bigram":   metric_srmse_bigram,
    "srmse_trigram":  metric_srmse_trigram,
    "srmse_quadgram": metric_srmse_quadgram,
}
