from click import command, option
from ananke_abm.models.traj_embed_updated.validate import gen_n_val_traj
import torch

@command()
@option("-ckpt", "--ckpt", type=str, required=True)
@option("-av", "--activities_csv", type=str, required=True)
@option("-pv", "--purposes_csv", type=str, required=True)
@option("-b", "--batch_size", type=int, default=32)
@option("-n", "--num_gen", type=int, default=20)
@option("-gp", "--gen_prefix", type=str, default="gen")
@option("-gc", "--gen_csv", type=str, default=None)
@option("-vc", "--val_csv", type=str, default=None)
@option("-es", "--eval_step_minutes", type=int, default=5)
@option("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
def gval_traj(ckpt, activities_csv, purposes_csv, batch_size, num_gen, gen_prefix, gen_csv, val_csv, eval_step_minutes, device):
    gen_n_val_traj(ckpt, activities_csv, purposes_csv, batch_size, num_gen, gen_prefix, gen_csv, val_csv, eval_step_minutes, device)
