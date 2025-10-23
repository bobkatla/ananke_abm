import os
import argparse
import torch
import yaml
import pandas as pd

from ananke_abm.benchmarks.contRNN.models.cont_rnn_vae import ContRNNVAE
from ananke_abm.benchmarks.contRNN.utils.io import ensure_dir

# uv run src/ananke_abm/benchmarks/contRNN/synth/synthesize.py --config-file src/ananke_abm/benchmarks/contRNN/configs/cont_rnn.yaml --ckpt src/output/cont_rnn_melb_full/best.pt --n-sample 500000 --encoded src/data/encoded/ --csv-out src/output/cont_rnn_melb_full/test.csv

def greedy_decode_batch(model, n, vocab, device, max_len):
    SOS = vocab["SOS"]
    EOS = vocab["EOS"]
    PAD = vocab.get("PAD", None)  # PAD may or may not exist

    z = torch.randn(n, model.latent_dim, device=device)
    h0, c0 = model.init_dec_state(z)

    def _step_input(act_ids, dur_scalar):
        emb = model.emb_dec(act_ids)  # (B,1,E)
        if dur_scalar.dim() == 1:
            dur3 = dur_scalar.view(-1, 1, 1)
        elif dur_scalar.dim() == 2:
            dur3 = dur_scalar.unsqueeze(-1)
        else:
            dur3 = dur_scalar
        return torch.cat([emb, dur3], dim=-1)  # (B,1,E+1)

    prev_act = torch.full((n, 1), SOS, dtype=torch.long, device=device)
    prev_dur = torch.zeros(n, 1, device=device)
    inputs = _step_input(prev_act, prev_dur)
    state = (h0, c0)

    acts = [prev_act]   # (B,1)
    durs = [prev_dur]   # (B,1)

    # Track which sequences have emitted EOS so we can stop updating them
    done = torch.zeros(n, dtype=torch.bool, device=device)

    for t in range(1, max_len):
        out, state = model.dec_rnn(inputs, state)   # (B,1,H)
        h_t = out[:, -1, :]                         # (B,H)
        logits_act = model.head_act(h_t)            # (B,V)
        logit_dur  = model.head_dur(h_t).squeeze(-1)  # (B,)

        # ----- MASK SPECIALS -----
        # never predict SOS after t=0
        logits_act[:, SOS] = float('-inf')
        # optionally prevent PAD entirely (recommended)
        if PAD is not None:
            logits_act[:, PAD] = float('-inf')
        # ensure at least one activity before EOS
        if t == 1:
            logits_act[:, EOS] = float('-inf')
        # sequences already done: force EOS to stay EOS (no changes)
        if done.any():
            # put large mass on EOS for done sequences
            logits_act[done, :] = float('-inf')
            logits_act[done, EOS] = 0.0
            # durations for done sequences set to ~0
            logit_dur[done] = -20.0  # sigmoid ~ ~2e-9

        next_act = torch.argmax(logits_act, dim=-1, keepdim=True)  # (B,1)
        next_dur = torch.sigmoid(logit_dur).unsqueeze(1)           # (B,1)

        acts.append(next_act)
        durs.append(next_dur)

        # mark EOS
        just_eos = (next_act.squeeze(1) == EOS)
        done |= just_eos

        # next input: for done seqs, keep feeding EOS with 0 dur so state stays stable
        feed_act = torch.where(done.view(-1,1), torch.full_like(next_act, EOS), next_act)
        feed_dur = torch.where(done.view(-1,1), torch.zeros_like(next_dur), next_dur)
        inputs = _step_input(feed_act, feed_dur)

        # early exit if all done
        if done.all():
            break

    acts_full = torch.cat(acts, dim=1)  # (B, L<=max_len)
    durs_full = torch.cat(durs, dim=1)
    return acts_full, durs_full

def strip_and_renorm(acts_row, durs_row, vocab, day_minutes=1800, min_eps=1e-6):
    SOS = vocab["SOS"]
    EOS = vocab["EOS"]
    PAD = vocab.get("PAD", None)

    keep_idx = []
    for i, a in enumerate(acts_row):
        if a == SOS or a == EOS:
            continue
        if PAD is not None and a == PAD:
            continue
        keep_idx.append(i)

    if len(keep_idx) == 0:
        return [], []  # nothing to keep

    a = [acts_row[i] for i in keep_idx]
    d = [float(durs_row[i]) for i in keep_idx]

    # clamp small negatives & floor tiny values to eps, then renorm
    d = [max(0.0, x) for x in d]
    s = sum(d)
    if s <= 0:
        d = [1.0 / len(d)] * len(d)
    else:
        d = [max(min_eps, x) for x in d]
        s = sum(d)
        d = [x / s for x in d]

    dur_min = [x * day_minutes for x in d]
    dur_min_round = [int(round(x)) for x in dur_min]
    diff = day_minutes - sum(dur_min_round)
    if diff != 0 and len(dur_min_round) > 0:
        dur_min_round[-1] += diff

    # remove any zero-length segments after rounding (rare but tidy)
    a2, dur2 = [], []
    for ai, dm in zip(a, dur_min_round):
        if dm > 0:
            a2.append(ai)
            dur2.append(dm)
    if len(a2) == 0:
        return [], []

    starts, t = [], 0
    for dm in dur2:
        starts.append(t)
        t += dm

    return a2, list(zip(starts, dur2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True, help="Path to config file (YAML format)")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (best.pt)")
    ap.add_argument("--n-sample", type=int, required=True)
    ap.add_argument("--encoded", default="data/encoded", help="Dir with vocab.json and meta.json")
    ap.add_argument("--csv-out", default="out/melb_schedules.csv", help="Output CSV path (apple-to-apple)")
    ap.add_argument("--day-minutes", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.csv_out) or ".")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=dev)
    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]
    rev_vocab = {i:s for s,i in vocab.items()}

    # Build a model skeleton and load weights
    # These dims must match training; update if you changed config
    cfg = yaml.safe_load(open(args.config_file))
    emb_dim=cfg["model"]["emb_dim"]
    rnn_hidden=cfg["model"]["rnn_hidden"]
    rnn_layers=cfg["model"]["rnn_layers"]
    latent_dim=cfg["model"]["latent_dim"]
    dropout=cfg["model"]["dropout"]

    model = ContRNNVAE(len(vocab), emb_dim, rnn_hidden, rnn_layers, latent_dim, dropout, max_len).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    remaining = args.n_sample
    batch = args.batch
    sample_base = 0

    # We synthesize persid as S00000001, S00000002, ...
    while remaining > 0:
        b = min(batch, remaining)
        with torch.no_grad():
            acts_full, durs_full = greedy_decode_batch(model, b, vocab, dev, max_len)
        acts_full = acts_full.cpu().tolist()
        durs_full = durs_full.cpu().tolist()

        for i in range(b):
            # strip specials, renorm, build minutes timeline
            a_ids, time_pairs = strip_and_renorm(acts_full[i], durs_full[i], vocab, day_minutes=args.day_minutes)
            if len(a_ids) == 0:
                continue  # rare degenerate; skip or resample
            # Build apple-to-apple rows
            persid = f"S{sample_base+i+1:08d}"
            hhid = ""  # unknown; keep blank or synthesize if needed
            stopno = 1
            for act_id, (start_min, dur_min) in zip(a_ids, time_pairs):
                purpose = rev_vocab[int(act_id)]
                startime = int(start_min)
                total_duration = int(dur_min)
                rows.append({
                    "persid": persid,
                    "hhid": hhid,
                    "stopno": stopno,
                    "purpose": purpose,
                    "startime": startime,
                    "total_duration": total_duration,
                })
                stopno += 1
            # (Optional) enforce end==day_minutes; our rounding already ensured that.
        sample_base += b
        remaining -= b

    # Write CSV
    df_out = pd.DataFrame(rows, columns=["persid","hhid","stopno","purpose","startime","total_duration"])
    df_out.to_csv(args.csv_out, index=False)
    print(f"Wrote {args.n_sample} synthetic schedules to {args.csv_out} "
          f"({len(df_out)} rows).")

if __name__ == "__main__":
    main()
