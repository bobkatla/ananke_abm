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
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False, num_workers=0, pin_memory=(dev.type=='cuda'))
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False, num_workers=0, pin_memory=(dev.type=='cuda'))

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
    scaler = torch.amp.GradScaler(enabled=(dev.type=='cuda'), device=dev.type)
    best_val = float("inf")
    best_epoch = -1
    out_dir = cfg["train"]["out_dir"]
    ensure_dir(out_dir)
    alpha, beta = lcfg["alpha"], lcfg["beta"]
    kl_warm = lcfg.get("kl_anneal_epochs", 0)
    warm_up_alpha = 15

    # --- setup AMP / scaler (CUDA only is simplest & robust) ---
    for epoch in range(1, tcfg["epochs"]+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Train {epoch}")
        for acts, durs, mask in pbar:
            acts = acts.to(dev, non_blocking=True)
            durs = durs.to(dev, non_blocking=True)
            mask = mask.to(dev, non_blocking=True)

            with torch.amp.autocast(enabled=(dev.type == 'cuda'), device_type=dev.type):
                logits_act, logits_dur, mu, logvar = model(acts, durs)
                # training: smoothing ON
                ce, mse, stats = masked_ce_mse(
                    logits_act, logits_dur, acts, durs, mask, label_smoothing=0.05
                )
                kl = kl_normal(mu, logvar)
                klw = beta * min(1.0, epoch/kl_warm) if kl_warm>0 else beta
                alpha_t = alpha * min(1.0, epoch / warm_up_alpha)
                loss = ce + alpha_t * mse + klw * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{ce.item():.3f}",
                mse=f"{mse.item():.3f}",
                alpha_t=f"{alpha_t:.1f}",
                klw=f"{klw:.3f}",
                kl=f"{kl.item():.3f}",
                n_tok=stats["n_tokens"]
            )

        # ----- validation -----
        model.eval()
        # Use full teacher forcing for a stable validation objective
        tf_train = model.tf
        model.tf = 1.0
        with torch.no_grad():
            v_loss = 0.0
            for acts, durs, mask in dl_va:
                acts = acts.to(dev, non_blocking=True)
                durs = durs.to(dev, non_blocking=True)
                mask = mask.to(dev, non_blocking=True)
                logits_act, logits_dur, mu, logvar = model(acts, durs)
                # validation: smoothing OFF; fixed α, β (paper-style selection)
                ce, mse, _ = masked_ce_mse(logits_act, logits_dur, acts, durs, mask, label_smoothing=0.0)
                kl = kl_normal(mu, logvar)
                loss = ce + alpha * mse + beta * kl
                v_loss += float(loss.item())
            v_loss /= max(1, len(dl_va))
        model.tf = tf_train
        
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
