import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ananke_abm.models.gen_schedule.utils.cfg import load_config, ensure_dir
from ananke_abm.models.gen_schedule.utils.seed import set_seed
from ananke_abm.models.gen_schedule.utils.ckpt import save_checkpoint
from ananke_abm.models.gen_schedule.models.factory import build_model
from ananke_abm.models.gen_schedule.losses.reg import time_total_variation
from ananke_abm.models.gen_schedule.losses.kl import kl_gaussian
from ananke_abm.models.gen_schedule.losses.home_loss import start_end_home_loss
from ananke_abm.models.gen_schedule.losses.utils_loss_pds import (
    loss_time_of_day_marginal,
    loss_presence_rate,
)

        
def train(config, output_dir, seed):
    cfg = load_config(config)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = output_dir
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "checkpoints"))
    ensure_dir(os.path.join(outdir, "plots"))

    data_npz_path = cfg["data"]["npz"]
    meta_path = data_npz_path.replace(".npz", "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)

    purpose_map = meta["purpose_map"]                  # {purpose_name: index}
    home_idx = purpose_map.get("Home", None)
    if home_idx is None:
        # fallback: most common first label in dataset
        tmp_ref = np.load(data_npz_path)["Y"].astype(np.int64)
        vals, counts = np.unique(tmp_ref[:, 0], return_counts=True)
        home_idx = int(vals[np.argmax(counts)])

    splits_path = cfg["data"]["split_pt"]
    split_obj = torch.load(splits_path, weights_only=False)
    train_dataset = split_obj["train_dataset"]
    val_dataset = split_obj["val_dataset"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg["train"]["batch_size"], max(1, len(train_dataset))),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg["train"]["batch_size"], max(1, len(val_dataset))),
        shuffle=False,
        drop_last=False,
    )

    model = build_model(cfg, meta).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val_loss = np.inf
    last_ckpt_path = os.path.join(outdir, "checkpoints", "last.pt")
    best_ckpt_path = os.path.join(outdir, "checkpoints", "best_val.pt")

    num_epochs = cfg["train"]["epochs"]
    min_epochs = cfg["train"]["min_epochs"]
    patience = cfg["train"]["patience"]
    warmup_epochs = int(max(1, num_epochs * cfg["train"]["beta_warm_frac"]))
    beta_target = cfg["train"]["beta_target"]

    lambda_tv = cfg["train"]["lambda_tv"]
    lambda_home = cfg["train"].get("lambda_home", 0.1)

    if cfg["model"]["method"] == "auto_pds":
        pds_npz = np.load(cfg["model"]["pds_path"])
        m_tod_emp = torch.tensor(
            pds_npz["m_tod"], dtype=torch.float32
        )                        # (P,T)
        presence_emp = torch.tensor(
            pds_npz["presence_rate"], dtype=torch.float32
        )                        # (P,)
        # move to device once
        m_tod_emp_PT = m_tod_emp.to(device)           # (P,T)
        presence_emp_P = presence_emp.to(device)      # (P,)
    else:
        m_tod_emp_PT = None
        presence_emp_P = None

    store_logs = []

    wait_epoch = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        beta = beta_target * min(1.0, epoch / max(1, warmup_epochs))

        total_train_loss = 0.0
        total_train_ce = 0.0
        total_train_kl = 0.0
        total_train_tv = 0.0
        total_train_home = 0.0
        num_train_batches = 0

        for batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch_labels = batch_labels.to(device)  # (B,T)
            logits_batch, mu, logvar = model(batch_labels)  # logits_batch: (B,T,P)

            ce_loss = F.cross_entropy(
                logits_batch.permute(0, 2, 1),  # (B,P,T)
                batch_labels,                   # (B,T)
                reduction="mean"
            )
            kl_loss = kl_gaussian(mu, logvar)
            tv_loss = time_total_variation(logits_batch)
            home_loss = start_end_home_loss(logits_batch, home_idx)

            loss = ce_loss + beta * kl_loss + lambda_tv * tv_loss + lambda_home * home_loss

            if cfg["model"]["method"] == "auto_pds":
                # m_tod_emp_PT and presence_emp_P were prepared once outside loop
                L_tod = loss_time_of_day_marginal(logits_batch, m_tod_emp_PT)
                L_presence = loss_presence_rate(logits_batch, presence_emp_P)

                loss = loss \
                    + cfg["train"]["lambda_tod"] * L_tod \
                    + cfg["train"]["lambda_presence"] * L_presence

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            total_train_loss += float(loss.item())
            total_train_ce += float(ce_loss.item())
            total_train_kl += float(kl_loss.item())
            total_train_tv += float(tv_loss.item())
            total_train_home += float(home_loss.item())
            num_train_batches += 1

        if num_train_batches > 0:
            avg_train_loss = total_train_loss / num_train_batches
            avg_train_ce = total_train_ce / num_train_batches
            avg_train_kl = total_train_kl / num_train_batches
            avg_train_tv = total_train_tv / num_train_batches
            avg_train_home = total_train_home / num_train_batches
        else:
            avg_train_loss = 0.0
            avg_train_ce = 0.0
            avg_train_kl = 0.0
            avg_train_tv = 0.0
            avg_train_home = 0.0

        model.eval()
        total_val_loss = 0.0
        total_val_ce = 0.0
        total_val_kl = 0.0
        total_val_tv = 0.0
        total_val_home = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                logits_batch, mu, logvar = model(batch_labels)

                ce_loss = F.cross_entropy(
                    logits_batch.permute(0, 2, 1),
                    batch_labels,
                    reduction="mean"
                )
                kl_loss = kl_gaussian(mu, logvar)
                tv_loss = time_total_variation(logits_batch)
                home_loss = start_end_home_loss(logits_batch, home_idx)

                val_loss = ce_loss + beta * kl_loss + lambda_tv * tv_loss + lambda_home * home_loss

                total_val_loss += float(val_loss.item())
                total_val_ce += float(ce_loss.item())
                total_val_kl += float(kl_loss.item())
                total_val_tv += float(tv_loss.item())
                total_val_home += float(home_loss.item())
                num_val_batches += 1

        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_ce = total_val_ce / num_val_batches
            avg_val_kl = total_val_kl / num_val_batches
            avg_val_tv = total_val_tv / num_val_batches
            avg_val_home = total_val_home / num_val_batches
        else:
            avg_val_loss = 0.0
            avg_val_ce = 0.0
            avg_val_kl = 0.0
            avg_val_tv = 0.0
            avg_val_home = 0.0

        save_checkpoint(
            {"model": model.state_dict(), "meta": meta, "cfg": cfg},
            last_ckpt_path,
        )
        wait_epoch += 1
        if epoch >= min_epochs and wait_epoch >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            break
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                {"model": model.state_dict(), "meta": meta, "cfg": cfg},
                best_ckpt_path,
            )
            wait_epoch = 0

        log_record = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": avg_train_loss,
            "train_ce": avg_train_ce,
            "train_kl": avg_train_kl,
            "train_tv": avg_train_tv,
            "train_home": avg_train_home,
            "val_loss": avg_val_loss,
            "val_ce": avg_val_ce,
            "val_kl": avg_val_kl,
            "val_tv": avg_val_tv,
            "val_home": avg_val_home,
            "num_train_batches": num_train_batches,
            "num_val_batches": num_val_batches,
        }
        store_logs.append(log_record)
    
    log_df = pd.DataFrame(store_logs)
    log_df.to_csv(os.path.join(outdir, "training_log.csv"), index=False)
