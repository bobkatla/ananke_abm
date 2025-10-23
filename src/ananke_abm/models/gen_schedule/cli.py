
import click
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ananke_abm.models.gen_schedule.utils.cfg import load_config, ensure_dir
from ananke_abm.models.gen_schedule.utils.seed import set_seed
from ananke_abm.models.gen_schedule.utils.ckpt import save_checkpoint
from ananke_abm.models.gen_schedule.dataio.rasterize import prepare_from_csv
from ananke_abm.models.gen_schedule.models.vae import ScheduleVAE, kl_gaussian
from ananke_abm.models.gen_schedule.losses.reg import time_total_variation

class GridDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.Y = d["Y"].astype(np.int64)
    def __len__(self): return self.Y.shape[0]
    def __getitem__(self, i):
        y = self.Y[i]
        return torch.from_numpy(y)

@click.group()
def main():
    pass

@main.command()
@click.option("--activities", type=click.Path(exists=True), required=True)
@click.option("--grid", type=int, default=10)
@click.option("--out", type=click.Path(), required=True)
def prepare(activities, grid, out):
    prepare_from_csv(activities, out, grid_min=grid)
    click.echo(f"Prepared grid at {out}")

@main.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="runs")
@click.option("--run", type=str, required=True)
@click.option("--seed", type=int, default=123)
def fit(config, output_dir, run, seed):
    cfg = load_config(config)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(output_dir, run)
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir,"checkpoints"))
    ensure_dir(os.path.join(outdir,"plots"))

    data_npz = cfg["data"]["npz"]
    meta_path = data_npz.replace(".npz", "_meta.json")
    meta = json.load(open(meta_path,"r"))
    P = len(meta["purpose_map"])
    L = meta["L"]

    ds = GridDataset(data_npz)
    n = len(ds)
    n_val = max(1, int(n*cfg["train"]["val_frac"]))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = ScheduleVAE(L=L, P=P, z_dim=cfg["model"]["z_dim"], emb_dim=cfg["model"]["emb_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = 1e9
    last_ckpt = os.path.join(outdir, "checkpoints", "last.pt")
    best_ckpt = os.path.join(outdir, "checkpoints", "best_val.pt")

    T = cfg["train"]["epochs"]
    warm = int(max(1, T * cfg["train"]["beta_warm_frac"]))
    beta_target = cfg["train"]["beta_target"]

    for epoch in range(1, T+1):
        model.train(); beta = beta_target * min(1.0, epoch/warm)
        train_loss = 0.0
        for y in tqdm(train_loader, desc=f"Epoch {epoch}/{T}"):
            y = y.to(device)  # (B,L)
            U, mu, logvar = model(y)
            ce = F.cross_entropy(U.permute(0,2,1), y)
            kl = kl_gaussian(mu, logvar)
            tv = time_total_variation(U)
            loss = ce + beta*kl + cfg["train"]["lambda_tv"]*tv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(1,len(train_loader))

        model.eval()
        with torch.no_grad():
            vloss, vce, vkl = 0.0, 0.0, 0.0
            for y in val_loader:
                y = y.to(device)
                U, mu, logvar = model(y)
                ce = F.cross_entropy(U.permute(0,2,1), y)
                kl = kl_gaussian(mu, logvar)
                loss = ce + beta*kl + cfg["train"]["lambda_tv"]*time_total_variation(U)
                vloss += float(loss.item())
                vce += float(ce.item())
                vkl += float(kl.item())
            if len(val_loader)>0:
                vloss /= len(val_loader)
                vce/=len(val_loader)
                vkl/=len(val_loader)

        save_checkpoint({"model": model.state_dict(), "meta": meta, "cfg": cfg}, last_ckpt)
        if vloss < best_val:
            best_val = vloss
            save_checkpoint({"model": model.state_dict(), "meta": meta, "cfg": cfg}, best_ckpt)

        log = {"epoch": epoch, "beta": beta, "train_loss": train_loss, "val_loss": vloss, "val_ce": vce, "val_kl": vkl}
        print(json.dumps(log))

    print("Done. Checkpoints saved:", best_ckpt, last_ckpt)

@main.command()
@click.option("--ckpt", type=click.Path(exists=True), required=True)
@click.option("--n", type=int, default=1000)
@click.option("--out", type=click.Path(), required=True)
def sample(ckpt, n, out):
    obj = torch.load(ckpt, map_location="cpu")
    cfg = obj["cfg"]
    meta = obj["meta"]
    P, L = len(meta["purpose_map"]), meta["L"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from ananke_abm.models.gen_schedule.models.vae import ScheduleVAE
    model = ScheduleVAE(L=L, P=P, z_dim=cfg["model"]["z_dim"], emb_dim=cfg["model"]["emb_dim"]).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    Z = torch.randn(n, cfg["model"]["z_dim"], device=device)
    with torch.no_grad():
        U = model.decoder(Z)
        y = torch.argmax(U, dim=-1).cpu().numpy().astype(np.int64)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, Y=y)
    print(f"Saved {n} samples to {out}")
