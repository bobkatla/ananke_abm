import os
import json
import click
import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

from ananke_abm.models.gen_schedule.models.crf.model import TransitionCRF


@click.command("train-crf")
@click.option("--cfg", "cfg_path", type=click.Path(exists=True), required=True,
              help="Path to crf_config.yaml")
def train_crf_cmd(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_npz    = cfg["crf"]["train_npz"]
    val_npz      = cfg["crf"]["val_npz"]
    save_path    = cfg["crf"]["save_path"]
    num_epochs   = cfg["crf"]["num_epochs"]
    batch_size   = cfg["crf"]["batch_size"]
    lr           = cfg["crf"]["lr"]
    weight_decay = cfg["crf"]["weight_decay"]
    log_every    = cfg["crf"].get("log_every", 10)

    # --- NEW: load meta to get home_idx ---
    meta_json_path = cfg["crf"]["meta_json"]
    with open(meta_json_path, "r", encoding="utf-8") as f:
        crf_meta = json.load(f)
    home_idx = int(crf_meta.get("home_idx", -1))
    if home_idx < 0:
        raise ValueError("home_idx not found in crf_meta.json; run prepare-crf-data again.")

    # load CRF training data produced by prepare-crf-data
    train_arr = np.load(train_npz)
    val_arr   = np.load(val_npz)

    U_train = torch.tensor(train_arr["U"])  # (N_train,T,P)
    Y_train = torch.tensor(train_arr["Y"])  # (N_train,T)
    U_val   = torch.tensor(val_arr["U"])    # (N_val,T,P)
    Y_val   = torch.tensor(val_arr["Y"])    # (N_val,T)

    _, T, P = U_train.shape

    train_ds = TensorDataset(U_train, Y_train)
    val_ds   = TensorDataset(U_val,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- NEW: pass home_idx (and keep a bias term for states) ---
    crf = TransitionCRF(num_purposes=P, home_idx=home_idx, use_bias=True).to(device)

    optimizer = optim.Adam(crf.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        crf.train()
        train_losses = []
        for (U_btp, Y_bt) in train_loader:
            U_btp = U_btp.to(device)  # (B,T,P)
            Y_bt  = Y_bt.to(device)   # (B,T)

            loss = crf.nll(U_btp, Y_bt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        crf.eval()
        val_losses = []
        with torch.no_grad():
            for (U_btp, Y_bt) in val_loader:
                U_btp = U_btp.to(device)
                Y_bt  = Y_bt.to(device)
                val_loss = crf.nll(U_btp, Y_bt)
                val_losses.append(val_loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val   = float(np.mean(val_losses))   if val_losses   else 0.0

        if epoch % log_every == 0 or epoch == 1 or epoch == num_epochs:
            msg = {"epoch": epoch, "train_nll": mean_train, "val_nll": mean_val}
            click.echo(json.dumps(msg))

        # best-by-val checkpoint
        if (best_val is None) or (mean_val < best_val):
            best_val = mean_val
            torch.save(
                {
                    "A_state_dict": crf.state_dict(),
                    "P": P,
                    "T": T,
                    "home_idx": home_idx,
                },
                save_path
            )

    click.echo(f"Saved best CRF to {save_path} with val_nll={best_val:.4f}")
