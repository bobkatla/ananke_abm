from ananke_abm.models.gen_schedule.compare.extract_metrics.metrics import METRIC_FUNCS
from ananke_abm.models.gen_schedule.compare.utils import load_reference, load_comparison_models, ensure_dir, assert_same_temporal_grid

import click


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------

@click.command("metric-tables")
@click.option("--ref_npz", type=click.Path(exists=True), required=True,
              help="Reference grid npz containing Y or Y_generated")
@click.option("--ref_meta", type=click.Path(exists=True), required=True,
              help="Reference meta json (purpose_map, grid_min, horizon_min)")
@click.option("--compare_dir", type=click.Path(exists=True), required=True,
              help="Directory with model .npz and matching meta json")
@click.option("--metrics", type=str, default="all",
              help="Comma-separated list of metrics to run, or 'all'. "
                   f"Available: {','.join(METRIC_FUNCS.keys())}")
@click.option("--outdir", type=click.Path(), required=True,
              help="Output directory to write all metric CSVs (later phases)")
def metric_tables_cli(ref_npz, ref_meta, compare_dir, metrics, outdir):
    """
    Phase 0: skeleton only.
    Loads reference + comparison models, parses metric list,
    then raises NotImplementedError for each metric.
    """
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

    # resolve metric list
    if metrics.strip().lower() == "all":
        metric_list = list(METRIC_FUNCS.keys())
    else:
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
        for m in metric_list:
            if m not in METRIC_FUNCS:
                raise ValueError(
                    f"Unknown metric '{m}'. Available: {list(METRIC_FUNCS.keys())}"
                )

    click.echo(f"[eval-tables] ref T={T_ref}, models={[m['name'] for m in models]}")
    click.echo(f"[eval-tables] metrics to run: {metric_list}")

    # call stubs (each will raise NotImplementedError for now)
    for mname in metric_list:
        func = METRIC_FUNCS[mname]
        click.echo(f"[eval-tables] running metric '{mname}' (stub)...")
        func(ref, models, outdir)
