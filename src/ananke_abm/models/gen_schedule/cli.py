import click
from ananke_abm.models.gen_schedule.models.pds import compute_pds_cli
from ananke_abm.models.gen_schedule.models.crf.cli_prepare import prepare_crf_data
from ananke_abm.models.gen_schedule.models.crf.cli_train import train_crf_cmd
from ananke_abm.models.gen_schedule.pipeline.comparisons import compare_samples


@click.group()
def main():
    pass


@main.command()
@click.option("--activities", type=click.Path(exists=True), required=True)
@click.option("--grid", type=int, default=10)
@click.option("--out", type=click.Path(), required=True)
@click.option("--val-frac", type=float, default=0.2)
@click.option("--seed", type=int, default=42)
def prepare(activities, grid, out, val_frac, seed):
    from ananke_abm.models.gen_schedule.dataio.rasterize import prepare_from_csv
    prepare_from_csv(activities, out, grid_min=grid, val_frac=val_frac, seed=seed)
    click.echo(f"Prepared grid at {out}")


@main.command("fit")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="runs")
@click.option("--seed", type=int, default=123)
def fit(config, output_dir, seed):
    from ananke_abm.models.gen_schedule.pipeline.train import train
    train(config, output_dir, seed)
    click.echo(f"Training complete in {output_dir}")


@main.command("sample-population")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=True,
              help="Trained checkpoint to sample from.")
@click.option("--num-samples", default=10000, show_default=True,
              help="How many synthetic individuals to generate.")
@click.option("--outprefix", type=click.Path(), required=True,
              help="Prefix for output files. Will emit <prefix>.npz, <prefix>_meta.json, <prefix>_preview.csv.")
@click.option("--seed", default=123, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--csv-max-persons", default=200, show_default=True,
              help="Max number of generated individuals to export into the human-readable preview CSV.")
@click.option("--decode-mode", type=click.Choice(["argmax", "crf"]), default="argmax", show_default=True,
              help="Decoding mode to convert model logits to discrete activity purposes.")
@click.option("--crf-path", type=click.Path(exists=True), default=None,
              help="Path to trained CRF model checkpoint (required if decode-mode is 'crf').")
@click.option("--enforce-nonhome", is_flag=True, default=False, show_default=True,
              help="If set with decode-mode 'crf', ensures at least one non-Home activity in each schedule.")
def sample_population(ckpt_path, num_samples, outprefix, seed, csv_max_persons, decode_mode, crf_path, enforce_nonhome):
    from ananke_abm.models.gen_schedule.pipeline.sample import sample
    sample(ckpt_path, num_samples, outprefix, seed, csv_max_persons, decode_mode, crf_path, enforce_nonhome)
    click.echo(f"Sampled {num_samples} individuals to {outprefix}.npz and related files.")


@main.command("eval-population")
@click.option("--samples", "samples_npz_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>.npz from sample-population.")
@click.option("--samples-meta", "samples_meta_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>_meta.json from sample-population.")
@click.option("--reference", "reference_grid_path", type=click.Path(exists=True), required=True,
              help="Reference prepared grid npz (e.g. runs/data/train_10min.npz).")
@click.option("--out-json", "out_json_path", type=click.Path(), required=True,
              help="Where to write metrics report JSON.")
def eval_population(samples_npz_path, samples_meta_path, reference_grid_path, out_json_path):
    from ananke_abm.models.gen_schedule.pipeline.eval import evaluate
    evaluate(samples_npz_path, samples_meta_path, reference_grid_path, out_json_path)
    click.echo(f"Evaluation complete. Report saved to {out_json_path}.")


@main.command("viz-population")
@click.option("--samples", "samples_npz_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>.npz from sample-population.")
@click.option("--samples-meta", "samples_meta_path", type=click.Path(exists=True), required=True,
              help="Output <prefix>_meta.json from sample-population.")
@click.option("--outdir", "outdir_path", type=click.Path(), required=True,
              help="Directory to write plots.")
@click.option("--reference", "reference_grid_path", type=click.Path(), default="",
              help="Optional reference prepared grid npz (real data) to overlay.")
@click.option("--not-use-logits", is_flag=True, default=True, show_default=True,
                help="If set, will not plot unaries using U_mean_logits and U_std_logits from the samples npz.")
def viz_population(samples_npz_path, samples_meta_path, outdir_path, reference_grid_path, not_use_logits):
    from ananke_abm.models.gen_schedule.pipeline.viz import visualize
    visualize(samples_npz_path, samples_meta_path, outdir_path, reference_grid_path, not_use_logits)
    click.echo(f"Visualization complete. Plots saved to {outdir_path}.")


main.add_command(compute_pds_cli)
main.add_command(prepare_crf_data)
main.add_command(train_crf_cmd)
main.add_command(compare_samples)