from click import command, option, Choice
import torch
from ananke_abm.models.traj_embed_updated.train import train_traj_embed


@command()
@option("-av", "--activities_csv", type=str, required=True)
@option("-pv", "--purposes_csv", type=str, required=True)
@option("--crf_mode", type=Choice(["linear", "semi"]), default="linear", help="CRF mode: linear aka frame-CRF or semi-CRF")
@option("-e", "--epochs", type=int, default=50)
@option("-b", "--batch_size", type=int, default=32)
@option("--lr", type=float, default=1e-3)
@option("--val_ratio", type=float, default=0.2)
@option("-o", "--outdir", type=str, default="./runs")
@option("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
def traj_embed(activities_csv, purposes_csv, crf_mode, epochs, batch_size, lr, val_ratio, outdir, device):
    """Train the TrajEmbed model."""
    train_traj_embed(activities_csv, purposes_csv, epochs, batch_size, lr, val_ratio, outdir, device, crf_mode)