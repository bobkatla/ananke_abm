
from ananke_abm.models.gen_schedule.compare.utils import load_reference, load_comparison_models, ensure_dir, assert_same_temporal_grid
from ananke_abm.models.gen_schedule.compare.viz_metrics.ToD import plot_tod_by_purpose
import click


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------

@click.command("plot-overview")
@click.option("--ref_npz", type=click.Path(exists=True), required=True,
              help="Reference grid npz containing Y or Y_generated")
@click.option("--ref_meta", type=click.Path(exists=True), required=True,
              help="Reference meta json (purpose_map, grid_min, horizon_min)")
@click.option("--compare_dir", type=click.Path(exists=True), required=True,
              help="Directory with model .npz and matching meta json")
@click.option("--outdir", type=click.Path(), required=True,
              help="Output directory to write all metric CSVs (later phases)")
def plot_overview(ref_npz, ref_meta, compare_dir, outdir):
    ensure_dir(outdir)

    # load data
    ref = load_reference(ref_npz, ref_meta)
    models = load_comparison_models(compare_dir)
    assert_same_temporal_grid(ref, models)

    # sanity: same T between ref and models
    T_ref = ref["Y"].shape[1]
    for m in models:
        T_m = m["Y"].shape[1]
        if T_m != T_ref:
            raise AssertionError(
                f"Time bins mismatch: ref has T={T_ref}, model {m['name']} has T={T_m}"
            )
    
    plot_tod_by_purpose(
        Y_list=[ref["Y"]] + [m["Y"] for m in models],
        dataset_names=["reference"] + [m["name"] for m in models],
        purpose_maps=[ref["purpose_map"]] + [m["purpose_map"] for m in models],
        time_grid=5,
        colors=None,
        start_time_min=0,
        outdir=outdir,
        show=False,
    )