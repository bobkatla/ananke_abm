import numpy as np
import os
import json
import click
from ananke_abm.models.gen_schedule.evals.report import make_report, save_report


def evaluate(samples_npz_path, samples_meta_path, reference_grid_path, out_json_path):    
    """
    Evaluate a previously sampled synthetic population against real data.
    No model or GPU required.
    """
    # load generated population
    synth_npz = np.load(samples_npz_path)
    generated_labels = synth_npz["Y_generated"].astype(np.int64)  # (N, L)

    # load metadata for purpose_map etc.
    with open(samples_meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)
    purpose_map = meta["purpose_map"]  # {purpose_name: index}

    # load reference (real) grid data
    ref_npz = np.load(reference_grid_path)
    reference_labels = ref_npz["Y"].astype(np.int64)
    # reference time-of-day marginals (precomputed in prepare())
    reference_tod_path = reference_grid_path.replace(".npz", "_tod.npy")
    reference_tod = np.load(reference_tod_path) if os.path.exists(reference_tod_path) else None

    # compute report
    report_dict = make_report(
        Y_synth=generated_labels,
        Y_ref=reference_labels,
        purpose_map=purpose_map,
        ref_tod=reference_tod,
    )
    save_report(report_dict, out_json_path)

    click.echo(json.dumps(report_dict, indent=2))
    click.echo(f"Saved metrics report to {out_json_path}")