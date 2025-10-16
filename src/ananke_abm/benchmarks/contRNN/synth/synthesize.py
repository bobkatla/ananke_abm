import os
import json
import argparse
import torch
import pandas as pd

from ananke_abm.benchmarks.contRNN.models.cont_rnn_vae import ContRNNVAE
from ananke_abm.benchmarks.contRNN.utils.io import ensure_dir, load_vocab

def greedy_decode_batch(model, n, vocab, device, max_len):
    """
    Paper-faithful greedy decoding:
      - start with SOS (dur=0)
      - unroll max_len steps
      - activities via argmax, durations via sigmoid
    Returns full sequences including specials so we can strip/trim per sample.
    """
    SOS = vocab["SOS"]; EOS = vocab["EOS"]
    z = torch.randn(n, model.latent_dim, device=device)

    # init decoder state
    h0, c0 = model.init_dec_state(z)

    # helper: make step input consistently 3D (B,1,E+1)
    def _step_input(act_ids, dur_scalar):
        # act_ids: (B,1) long; dur_scalar: (B,1) or (B,) float -> (B,1,1)
        emb = model.emb_dec(act_ids)                         # (B,1,E)
        if dur_scalar.dim() == 1:                            # (B,)
            dur3 = dur_scalar.view(-1, 1, 1)                 # (B,1,1)
        elif dur_scalar.dim() == 2:                          # (B,1)
            dur3 = dur_scalar.unsqueeze(-1)                  # (B,1,1)
        else:                                                # already (B,1,1)
            dur3 = dur_scalar
        return torch.cat([emb, dur3], dim=-1)                # (B,1,E+1)

    # first input is SOS with duration 0
    prev_act = torch.full((n, 1), SOS, dtype=torch.long, device=device)
    prev_dur = torch.zeros(n, 1, device=device)              # (B,1) -> _step_input will make (B,1,1)
    inputs = _step_input(prev_act, prev_dur)
    state = (h0, c0)

    acts_steps = [prev_act]    # list of (B,1) long
    durs_steps = [prev_dur]    # list of (B,1) float

    for t in range(1, max_len):
        out, state = model.dec_rnn(inputs, state)            # (B,1,H)
        h_t = out[:, -1, :]                                  # (B,H)
        logits_act = model.head_act(h_t)                     # (B,V)
        logit_dur  = model.head_dur(h_t).squeeze(-1)         # (B,)

        next_act = torch.argmax(logits_act, dim=-1, keepdim=True)  # (B,1)
        next_dur = torch.sigmoid(logit_dur).unsqueeze(1)            # (B,1)

        acts_steps.append(next_act)
        durs_steps.append(next_dur)

        inputs = _step_input(next_act, next_dur)

    acts_full = torch.cat(acts_steps, dim=1)   # (B, L)
    durs_full = torch.cat(durs_steps, dim=1)   # (B, L)
    return acts_full, durs_full

def strip_and_renorm(acts_row, durs_row, vocab, day_minutes=1800):
    """
    - Remove SOS/EOS tokens
    - Renormalize durations to sum=1
    - Convert to minutes
    - Build start times (0..day_minutes)
    """
    SOS = vocab["SOS"]; EOS = vocab["EOS"]
    # strip specials
    keep_idx = [i for i, a in enumerate(acts_row) if a not in (SOS, EOS)]
    if len(keep_idx) == 0:
        return [], []  # degenerate; caller may resample if desired
    a = [acts_row[i] for i in keep_idx]
    d = [float(durs_row[i]) for i in keep_idx]
    # clamp and renorm
    d = [max(0.0, x) for x in d]
    s = sum(d)
    if s <= 0:
        # fallback: evenly split
        d = [1.0/len(d)] * len(d)
    else:
        d = [x / s for x in d]
    # minutes and timeline
    dur_min = [x * day_minutes for x in d]
    # round to nearest minute and fix residual on last segment
    dur_min_round = [int(round(x)) for x in dur_min]
    diff = day_minutes - sum(dur_min_round)
    if diff != 0 and len(dur_min_round) > 0:
        dur_min_round[-1] += diff
    # build starts
    start = 0
    starts = []
    for i, dm in enumerate(dur_min_round):
        starts.append(start)
        start += dm
    return a, list(zip(starts, dur_min_round))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (best.pt)")
    ap.add_argument("--n-sample", type=int, required=True)
    ap.add_argument("--encoded", default="data/encoded", help="Dir with vocab.json and meta.json")
    ap.add_argument("--csv-out", default="out/melb_schedules.csv", help="Output CSV path (apple-to-apple)")
    ap.add_argument("--day-minutes", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.csv_out) or ".")
    vocab = load_vocab(os.path.join(args.encoded, "vocab.json"))
    meta  = json.load(open(os.path.join(args.encoded, "meta.json")))
    rev_vocab = {i:s for s,i in vocab.items()}
    max_len = meta["max_len"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a model skeleton and load weights
    # These dims must match training; update if you changed config
    emb_dim=256; rnn_hidden=256; rnn_layers=4; latent_dim=6; dropout=0.1
    model = ContRNNVAE(len(vocab), emb_dim, rnn_hidden, rnn_layers, latent_dim, dropout, max_len).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt["model"]); model.eval()

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
            t = 0
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
