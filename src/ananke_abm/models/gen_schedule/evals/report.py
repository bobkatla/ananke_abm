import json
import os
import numpy as np
from ananke_abm.models.gen_schedule.evals.metrics import minutes_share, tod_marginals, bigram_matrix, l1_distance
from ananke_abm.models.gen_schedule.losses.jsd import jsd

def compute_all_home_rate(Y, home_idx):
    return float(np.mean((Y==home_idx).all(axis=1)))

def start_end_home_stats(Y, home_idx):
    start = float(np.mean(Y[:,0]==home_idx))
    end = float(np.mean(Y[:,-1]==home_idx))
    return start, end

def diversity_ratio(Y):
    seen = set()
    for row in Y:
        seen.add(row.tobytes())
    return float(len(seen))/float(len(Y))

def make_report(Y_synth, Y_ref, purpose_map, ref_tod=None):
    P = len(purpose_map); L = Y_synth.shape[1]
    home_idx = purpose_map.get("Home", None)
    if home_idx is None:
        vals, counts = np.unique(Y_ref[:,0], return_counts=True)
        home_idx = int(vals[np.argmax(counts)])

    share_syn = minutes_share(Y_synth, P)
    share_ref = minutes_share(Y_ref, P)
    share_ae = np.abs(share_syn - share_ref)

    m_syn = tod_marginals(Y_synth, P)
    m_ref = ref_tod if ref_tod is not None else tod_marginals(Y_ref, P)

    B_syn = bigram_matrix(Y_synth, P)
    B_ref = bigram_matrix(Y_ref, P)
    bigram_L1 = l1_distance(B_syn, B_ref)

    all_home = compute_all_home_rate(Y_synth, home_idx)
    start_home, end_home = start_end_home_stats(Y_synth, home_idx)
    div = diversity_ratio(Y_synth)

    jsds = [jsd(m_ref[t], m_syn[t]) for t in range(L)]
    report = {
        "P": P, "L": int(L),
        "home_idx": int(home_idx),
        "minutes_share": {"synth": share_syn.tolist(), "ref": share_ref.tolist(), "abs_error": share_ae.tolist()},
        "bigram": {"L1": bigram_L1},
        "tod_jsd_macro": float(np.mean(jsds)),
        "all_home_rate": all_home,
        "start_home_rate": start_home,
        "end_home_rate": end_home,
        "diversity_ratio": div,
    }
    return report

def save_report(report, out_json):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
