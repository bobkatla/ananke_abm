
from ananke_abm.models.gen_schedule.compare.utils import (
    load_reference,
    load_comparison_models,
    ensure_dir,
    assert_same_temporal_grid,
    schedule_counts,
    ngram_counts,
)
from ananke_abm.models.gen_schedule.compare.viz_metrics.ToD import plot_tod_by_purpose
from ananke_abm.models.gen_schedule.compare.viz_metrics.duration import plot_duration_boxplots
from ananke_abm.models.gen_schedule.compare.viz_metrics.lorenz import plot_lorenz_for_models
import numpy as np
import click


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------

@click.command("plot-overview")
@click.option("--ref_npz", type=click.Path(exists=True), required=True,
              help="Reference grid npz containing Y or Y_generated")
@click.option("--ref_meta", type=click.Path(exists=True), required=True,
              help="Reference meta json (purpose_map, grid_min, horizon_min)")
@click.option("--train_npz", type=click.Path(exists=True), required=True,
              help="Training grid npz containing Y or Y_generated")
@click.option("--train_meta", type=click.Path(exists=True), required=True,
              help="Training meta json (purpose_map, grid_min, horizon_min)")
@click.option("--compare_dir", type=click.Path(exists=True), required=True,
              help="Directory with model .npz and matching meta json")
@click.option("--outdir", type=click.Path(), required=True,
              help="Output directory to write all metric CSVs (later phases)")
def plot_overview(ref_npz, ref_meta, train_npz, train_meta, compare_dir, outdir):
    ensure_dir(outdir)

    # load data
    ref = load_reference(ref_npz, ref_meta)
    models = load_comparison_models(compare_dir)
    train_data = load_reference(train_npz, train_meta)
    assert_same_temporal_grid(ref, models)

    predefined_colors = {
        "reference": "black",
        "training": "gray",
        "VAE_CNN": "blue",
        "VAE_CNN_CRF": "orange",
        "VAE_CNN_CRF_rejection": "green",
        "VAE_CNN_CRF_constrained": "red",
        "ContRNN": "purple",
    }

    # sanity: same T between ref and models
    T_ref = ref["Y"].shape[1]
    for m in models:
        T_m = m["Y"].shape[1]
        if T_m != T_ref:
            raise AssertionError(
                f"Time bins mismatch: ref has T={T_ref}, model {m['name']} has T={T_m}"
            )
        
    counts_ref = np.array(list(schedule_counts(ref["Y"]).values()), dtype=np.float64)
    to_plot_models = {"Reference": counts_ref}
    for m in models:
        counts_syn = np.array(list(schedule_counts(m["Y"]).values()), dtype=np.float64)
        to_plot_models[m["name"]] = counts_syn

    plot_lorenz_for_models(
        model_counts=to_plot_models,
        title="",
        output_dir=outdir,
        show=False,
        prefix="models_compare",
        colors=predefined_colors
    )
        
    # plot_duration_boxplots(
    #     Y_list=[ref["Y"]] + [m["Y"] for m in models],
    #     dataset_names=["Reference"] + [m["name"] for m in models],
    #     purpose_maps=[ref["purpose_map"]] + [m["purpose_map"] for m in models],
    #     colors=[predefined_colors.get("reference", "black")] +
    #            [predefined_colors.get(m["name"], None) for m in models],
    #     output_dir=outdir,
    #     show=False,
    #     prefix="models_compare",
    #     layout="separate"
    # )

    # plot_duration_boxplots(
    #     Y_list=[ref["Y"], train_data["Y"]],
    #     dataset_names=["Reference", "Sample"],
    #     purpose_maps=[ref["purpose_map"], train_data["purpose_map"]],
    #     colors=[predefined_colors["reference"], predefined_colors["training"]],
    #     output_dir=outdir,
    #     show=False,
    #     prefix="ref_vs_train",
    #     layout="compressed"
    # )
    
    # plot_tod_by_purpose(
    #     Y_list=[ref["Y"]] + [m["Y"] for m in models],
    #     dataset_names=["Reference"] + [m["name"] for m in models],
    #     purpose_maps=[ref["purpose_map"]] + [m["purpose_map"] for m in models],
    #     time_grid=5,
    #     colors=[predefined_colors.get("reference", "black")] +
    #            [predefined_colors.get(m["name"], None) for m in models],
    #     start_time_min=0,
    #     outdir=outdir,
    #     show=False,
    #     prefix="models_compare"
    # )

    # plot_tod_by_purpose(
    #     Y_list=[ref["Y"], train_data["Y"]],
    #     dataset_names=["Reference", "Sample"],
    #     purpose_maps=[ref["purpose_map"], train_data["purpose_map"]],
    #     time_grid=5,
    #     colors=[predefined_colors["reference"], predefined_colors["training"]],
    #     start_time_min=0,
    #     outdir=outdir,
    #     show=False,
    #     prefix="ref_vs_train"
    # )