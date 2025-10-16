import os, json, hashlib
import numpy as np
import pandas as pd

REQUIRED = ["persid","hhid","stopno","purpose","startime","total_duration"]

def _stable_u(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(16**8)

def split_by_persid(persids, splits=(0.8,0.1,0.1)):
    u = np.array([_stable_u(p) for p in persids])
    c1, c2 = splits[0], splits[0]+splits[1]
    i_tr = np.where(u < c1)[0]
    i_va = np.where((u>=c1)&(u<c2))[0]
    i_te = np.where(u>=c2)[0]
    return i_tr, i_va, i_te

def build_vocab(purposes: pd.Series):
    uniq = sorted(purposes.unique().tolist())
    vocab = {"SOS": 0, "EOS": 1}
    for i, p in enumerate(uniq, start=2):
        vocab[p] = i
    inv_vocab = {i:s for s,i in vocab.items()}
    return vocab, inv_vocab

def prepare_from_csv(csv_path: str, out_dir: str, max_len=20, day_minutes=1800, splits=(0.8,0.1,0.1)):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).sort_values(["persid","stopno"]).copy()
    assert set(REQUIRED).issubset(df.columns), f"CSV must contain {REQUIRED}"
    # integrity checks
    df["endtime"] = df["startime"] + df["total_duration"]
    per_person_total = df.groupby("persid")["total_duration"].sum()
    assert np.allclose(per_person_total.values, day_minutes, atol=1e-6), "Σ duration != day_minutes"
    first_p = df.groupby("persid").first()["purpose"]
    last_p  = df.groupby("persid").last()["purpose"]
    assert (first_p == "Home").all() and (last_p == "Home").all(), "Not home-bounded for all"
    def contiguous_ok(g):
        g = g.sort_values("stopno")
        return np.allclose(g["startime"].values[1:], (g["startime"]+g["total_duration"]).values[:-1], atol=1e-6)
    ok = df.groupby("persid").apply(contiguous_ok).values
    assert ok.all(), "Gaps/overlaps exist"

    # collapse consecutive identical purposes
    df["purpose_shift"] = df.groupby("persid")["purpose"].shift()
    df["new_group"] = (df["purpose"] != df["purpose_shift"]).astype(int)
    df["grp_id"] = df.groupby("persid")["new_group"].cumsum()
    collapsed = (df.groupby(["persid","grp_id","purpose"], as_index=False)
                   .agg(startime=("startime","first"),
                        total_duration=("total_duration","sum")))

    last_end = (collapsed["startime"] + collapsed["total_duration"]).groupby(collapsed["persid"]).max()
    assert np.allclose(last_end.values, day_minutes, atol=1e-6), "Collapsed does not end at day length"

    vocab, inv_vocab = build_vocab(collapsed["purpose"])
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    def build_sequence(rows):
        acts = [vocab["SOS"]]; durs = [0.0]
        for p, dur in zip(rows["purpose"].tolist(), rows["total_duration"].tolist()):
            acts.append(vocab[p]); durs.append(float(dur)/float(day_minutes))
        acts.append(vocab["EOS"]); durs.append(0.0)
        if len(acts) > max_len:
            acts = acts[:max_len]; durs = durs[:max_len]
            acts[-1] = vocab["EOS"]; durs[-1] = 0.0
        else:
            pad = max_len - len(acts)
            acts.extend([vocab["EOS"]]*pad); durs.extend([0.0]*pad)
        # mask: True on non-specials
        m = np.array([a not in (vocab["SOS"], vocab["EOS"]) for a in acts], dtype=bool)
        # renorm inside mask to be safe
        arr = np.array(durs, dtype=np.float64)
        s = arr[m].sum()
        if s > 0 and not np.isclose(s, 1.0, atol=1e-6):
            arr[m] = arr[m] / s
            durs = arr.tolist()
        return acts, durs, m.astype(np.uint8).tolist()

    acts_list, durs_list, mask_list, pid_list = [], [], [], []
    for pid, g in collapsed.groupby("persid", sort=False):
        a, d, m = build_sequence(g)
        acts_list.append(a); durs_list.append(d); mask_list.append(m); pid_list.append(pid)
    acts = np.array(acts_list, dtype=np.int64)
    durs = np.array(durs_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.uint8).astype(bool)
    persid = np.array(pid_list, dtype=object)

    # splits
    idx_tr, idx_va, idx_te = split_by_persid(persid, splits)
    def take(idx): return acts[idx], durs[idx], mask[idx], persid[idx]

    a_tr, d_tr, m_tr, p_tr = take(idx_tr)
    a_va, d_va, m_va, p_va = take(idx_va)
    a_te, d_te, m_te, p_te = take(idx_te)

    # save (torch first, fallback to npz)
    def save(base, a,d,m,p):
        try:
            import torch
            obj = {"acts": torch.as_tensor(a), "durs": torch.as_tensor(d),
                   "mask": torch.as_tensor(m), "persid": list(map(str,p))}
            path = base+".pt"; torch.save(obj, path); return path
        except Exception:
            path = base+".npz"; np.savez_compressed(path, acts=a, durs=d, mask=m, persid=p); return path

    paths = {
        "train": save(os.path.join(out_dir, "train"), a_tr,d_tr,m_tr,p_tr),
        "val":   save(os.path.join(out_dir, "val"),   a_va,d_va,m_va,p_va),
        "test":  save(os.path.join(out_dir, "test"),  a_te,d_te,m_te,p_te),
    }
    meta = {
        "max_len": max_len, "day_minutes": day_minutes,
        "splits": {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
        "vocab_size": len(vocab), "vocab_keys": list(vocab.keys())
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"paths": paths, "meta": meta, "vocab": vocab}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/encoded")
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--day-minutes", type=int, default=1800)
    ap.add_argument("--splits", nargs=3, type=float, default=[0.8,0.1,0.1])
    args = ap.parse_args()
    info = prepare_from_csv(args.csv, args.out, args.max_len, args.day_minutes, tuple(args.splits))
    print(info["paths"])
