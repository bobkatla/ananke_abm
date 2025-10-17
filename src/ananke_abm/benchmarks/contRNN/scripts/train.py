import os
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ananke_abm.benchmarks.contRNN.data_process.prepare_from_csv import prepare_from_csv
from ananke_abm.benchmarks.contRNN.models.cont_rnn_vae import ContRNNVAE
from ananke_abm.benchmarks.contRNN.losses.schedule_losses import masked_ce_mse, kl_normal
from ananke_abm.benchmarks.contRNN.utils.seeds import set_all
from ananke_abm.benchmarks.contRNN.utils.io import load_encoded, ensure_dir, save_ckpt

def main(cfg_path="configs/cont_rnn.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    set_all(cfg["seed"])
    # Phase A under the hood (so you can pass raw CSV directly)
    process_data_paths = prepare_from_csv(
        cfg["data"]["csv_path"], cfg["data"]["out_dir"],
        cfg["data"]["max_len"], cfg["data"]["day_minutes"], tuple(cfg["data"]["splits"])
    )
    train = load_encoded(process_data_paths["paths"]["train"])
    val   = load_encoded(process_data_paths["paths"]["val"])
    vocab = process_data_paths["vocab"]
    max_len = cfg["data"]["max_len"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = TensorDataset(train["acts"], train["durs"], train["mask"])
    ds_va = TensorDataset(val["acts"],   val["durs"],   val["mask"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False)

    mcfg = cfg["model"]
    tcfg = cfg["train"]
    lcfg = cfg["loss"]
    model = ContRNNVAE(
        vocab_size=len(vocab),
        emb_dim=mcfg["emb_dim"], rnn_hidden=mcfg["rnn_hidden"], rnn_layers=mcfg["rnn_layers"],
        latent_dim=mcfg["latent_dim"], dropout=mcfg["dropout"], max_len=max_len,
        teacher_forcing=tcfg["teacher_forcing"]
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    best_val = float("inf")
    best_epoch = -1
    out_dir = cfg["train"]["out_dir"]
    ensure_dir(out_dir)
    alpha, beta = lcfg["alpha"], lcfg["beta"]
    kl_warm = lcfg.get("kl_anneal_epochs", 0)

    for epoch in range(1, tcfg["epochs"]+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Train {epoch}")
        loss_sum = 0.0
        for acts, durs, mask in pbar:
            acts = acts.to(dev)
            durs = durs.to(dev)
            mask = mask.to(dev)
            logits_act, logits_dur, mu, logvar = model(acts, durs)
            ce, mse = masked_ce_mse(logits_act, logits_dur, acts, durs, mask)
            kl = kl_normal(mu, logvar)
            klw = beta * min(1.0, epoch/kl_warm) if kl_warm>0 else beta
            loss = ce + alpha*mse + klw*kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{ce.item():.3f}", mse=f"{mse.item():.3f}", kl=f"{kl.item():.3f}")

        # val
        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            for acts, durs, mask in dl_va:
                acts = acts.to(dev); durs = durs.to(dev); mask = mask.to(dev)
                logits_act, logits_dur, mu, logvar = model(acts, durs)
                ce, mse = masked_ce_mse(logits_act, logits_dur, acts, durs, mask)
                kl = kl_normal(mu, logvar)
                klw = beta
                loss = ce + alpha*mse + klw*kl
                v_loss += loss.item()
            v_loss /= max(1, len(dl_va))

        if v_loss < best_val:
            print(f"Epoch {epoch}: val loss improved {best_val:.3f} -> {v_loss:.3f}")
            best_val = v_loss
            best_epoch = epoch
            save_ckpt(os.path.join(out_dir,"best.pt"), model, opt, epoch, vocab, max_len, best=True)
        save_ckpt(os.path.join(out_dir,"last.pt"), model, opt, epoch, vocab, max_len, best=False)

        # simple early stopping
        if epoch - best_epoch > cfg["train"]["patience"]:
            print(f"Early stop at epoch {epoch} (best={best_epoch}, val={best_val:.3f})")
            break

    print(f"Done. Best epoch {best_epoch}, best val {best_val:.3f}")

if __name__ == "__main__":
    main(cfg_path="src/ananke_abm/benchmarks/contRNN/configs/cont_rnn.yaml")
