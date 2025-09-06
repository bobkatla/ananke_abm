# src/dataio/mockgen_fixed.py
# Deterministic mock generator for weekday activity schedules (purpose + time only)
# Outputs: persons.csv, schedules.csv, purposes.csv
# Guarantees:
# - Day starts at 0.0 with "home" and ends at 24.0 with "home"
# - Segments sorted; no overlaps
# - Deterministic given seed

import argparse
import csv
import pathlib
from dataclasses import dataclass
import numpy as np

DEFAULT_SEED = 12345

EMPLOY_CATS = np.array(["fulltime", "parttime", "student", "unemployed", "retired"])
EMPLOY_P    = np.array([0.55, 0.20, 0.15, 0.05, 0.05])

PURPOSES = ["home","work","lunch","shopping","gym","errand","leisure"]

PURPOSE_FEATURES = {
    "home":     dict(importance=0.9,  flexibility=0.8, start_mu=0.0,  start_std=6.0, dur_mu=12.0, dur_std=4.0, category="maintenance",   skip_prob=0.0),
    "work":     dict(importance=0.95, flexibility=0.2, start_mu=9.0,  start_std=1.5, dur_mu=7.0,  dur_std=1.5, category="mandatory",     skip_prob=0.1),
    "lunch":    dict(importance=0.6,  flexibility=0.4, start_mu=12.5, start_std=0.7, dur_mu=1.0,  dur_std=0.3, category="maintenance",   skip_prob=0.05),
    "shopping": dict(importance=0.3,  flexibility=0.7, start_mu=18.5, start_std=2.0, dur_mu=0.8,  dur_std=0.4, category="discretionary", skip_prob=0.7),
    "gym":      dict(importance=0.4,  flexibility=0.6, start_mu=19.5, start_std=1.8, dur_mu=1.0,  dur_std=0.4, category="discretionary", skip_prob=0.6),
    "errand":   dict(importance=0.35, flexibility=0.6, start_mu=17.0, start_std=2.5, dur_mu=0.7,  dur_std=0.4, category="maintenance",   skip_prob=0.6),
    "leisure":  dict(importance=0.5,  flexibility=0.8, start_mu=20.0, start_std=1.8, dur_mu=1.2,  dur_std=0.6, category="discretionary", skip_prob=0.2),
}

@dataclass
class Profile:
    name: str
    mu:   tuple  # (start_work, lunch_time, work1_len, work2_len)
    sig:  tuple
    p_shop: float
    p_gym:  float

PROFILES = [
    Profile("standard_9_5", mu=(8.5, 12.3, 3.5, 3.5), sig=(0.5, 0.2, 0.6, 0.6), p_shop=0.30, p_gym=0.25),
    Profile("late_shift",   mu=(11.0, 15.0, 4.0, 4.0), sig=(0.6, 0.5, 0.8, 0.8), p_shop=0.25, p_gym=0.20),
    Profile("flex_part",    mu=(9.5, 12.5, 3.0, 2.0),  sig=(0.8, 0.5, 0.8, 0.8), p_shop=0.35, p_gym=0.30),
]
PROFILE_P = np.array([0.60, 0.25, 0.15])

def _clip_int(x, lo, hi):
    return int(max(lo, min(hi, round(x))))

def _normal_pos(rng, mu, sig, lo, hi):
    # truncated normal by rejection
    for _ in range(1000):
        v = rng.normal(mu, sig)
        if lo <= v <= hi:
            return float(v)
    return float(np.clip(rng.normal(mu, sig), lo, hi))

def _choose_profile(rng) -> Profile:
    idx = rng.choice(len(PROFILES), p=PROFILE_P)
    return PROFILES[idx]

def _person_row(rng, pid_str: str):
    age = _clip_int(rng.normal(40, 12), 18, 75)
    emp = rng.choice(EMPLOY_CATS, p=EMPLOY_P)
    hh = _clip_int(rng.normal(2.6, 1.0), 1, 6)
    kids = max(0, _clip_int(rng.normal(0.7, 1.0), -2, 5))
    cars = max(0, _clip_int(rng.normal(1.0, 0.7), -1, 3))
    zone = f"Z{int(rng.integers(1, 101))}"
    return [pid_str, age, emp, hh, kids, cars, zone]

def _build_day_segments(rng):
    """
    Draft a plausible weekday (may include overlaps/gaps before sanitize).
    We'll sanitize + enforce home-bound later.
    """
    prof = _choose_profile(rng)
    s_work  = _normal_pos(rng, prof.mu[0], prof.sig[0], 5.0, 12.5)
    lunch   = _normal_pos(rng, prof.mu[1], prof.sig[1], 11.0, 16.0)
    w1      = max(0.4, rng.normal(prof.mu[2], prof.sig[2]))
    w2      = max(0.4, rng.normal(prof.mu[3], prof.sig[3]))

    segs = []
    # Start-of-day: home placeholder (we’ll force exact start later)
    segs.append(("home", 0.0, max(0.3, s_work - 0.3)))

    # Work 1
    segs.append(("work", s_work, w1))
    # Lunch
    lunch_len = float(np.clip(rng.normal(1.0, 0.2), 0.5, 1.5))
    segs.append(("lunch", lunch, lunch_len))
    # Work 2
    s2 = lunch + lunch_len
    segs.append(("work", s2, w2))

    # Optional shopping (use the chosen profile!)
    if rng.random() < prof.p_shop:
        segs.append(("shopping", s2 + rng.uniform(0.1, 0.6),
                     float(np.clip(rng.normal(0.7, 0.3), 0.3, 1.8))))
    # Optional gym (use the chosen profile!)
    if rng.random() < prof.p_gym:
        latest = max([s2 + w2] + [s + d for (p, s, d) in segs if p == "shopping"])
        segs.append(("gym", latest + rng.uniform(0.1, 0.6),
                     float(np.clip(rng.normal(1.0, 0.3), 0.5, 2.0))))

    # Evening leisure
    eve_start = float(np.clip(rng.normal(19.3, 0.7), 17.5, 21.5))
    segs.append(("leisure", eve_start,
                 float(np.clip(rng.normal(1.0, 0.4), 0.4, 2.5))))

    # Final home placeholder; will be adjusted to end at 24 exactly
    segs.append(("home", max(eve_start + 1.0, s2 + w2), 0.5))
    return segs

# --- add these helpers near the top ---
SCALE = 1000            # ticks per hour (0.001 h resolution)
DAY_END = 24 * SCALE    # 24000 ticks
MIN_SEG_TICKS = int(0.2 * SCALE)  # minimum duration for non-fill segments (~0.2h)

def _to_ticks(x: float) -> int:
    # round half up to the nearest tick
    return int(round(x * SCALE))

def _from_ticks(t: int) -> float:
    return t / SCALE

def _sanitize_and_sort(segs):
    """
    Sanitize a list of (purpose, start_hour, dur_hour) into a non-overlapping,
    home-bound day [0,24] using integer ticks (0.001h).
    Returns list of (purpose, start_hour_float_3dp, dur_hour_float_3dp).
    """
    # 1) to ticks and sort by proposed start
    segs_t = []
    for (p, s, d) in sorted(segs, key=lambda x: x[1]):
        s_t = max(0, min(DAY_END, _to_ticks(s)))
        d_t = max(MIN_SEG_TICKS, _to_ticks(d))  # initial min dur; clamped below
        segs_t.append((p, s_t, d_t))

    # 2) enforce no overlaps (allow gaps), clamp to day
    clean = []
    tcur = 0
    for (p, s_t, d_t) in segs_t:
        s_t = max(s_t, tcur)                # no overlap
        if s_t >= DAY_END:
            break
        d_t = max(MIN_SEG_TICKS, min(d_t, DAY_END - s_t))
        clean.append((p, s_t, d_t))
        tcur = s_t + d_t
        if tcur >= DAY_END:
            break

    # 3) ensure the day exists
    if not clean:
        clean = [("home", 0, DAY_END)]

    # 4) ensure first block is exactly home@0 (insert or relabel)
    first_p, first_s, first_d = clean[0]
    if first_s > 0:
        # insert filler home from 0 up to first start (no min duration here)
        clean.insert(0, ("home", 0, min(first_s, DAY_END)))
    elif first_p != "home":
        clean[0] = ("home", 0, first_d)
    else:
        clean[0] = (first_p, 0, first_d)

    # 5) merge consecutive identical purposes (exact in ticks)
    merged = []
    for (p, s_t, d_t) in clean:
        if merged and merged[-1][0] == p:
            p0, s0, d0 = merged[-1]
            merged[-1] = (p0, s0, d0 + d_t)
        else:
            merged.append((p, s_t, d_t))

    # 6) FINAL SNAP: unconditionally make the last block 'home' and end at 24h
    pL, sL, dL = merged[-1]
    endL = sL + dL
    if pL != "home":
        if endL < DAY_END:
            merged.append(("home", endL, DAY_END - endL))
        else:
            # ends exactly at 24 but last isn't home -> relabel last as home
            merged[-1] = ("home", sL, dL)
    # re-read last and force end at DAY_END
    pL, sL, dL = merged[-1]
    if sL > DAY_END:  # shouldn’t happen; clamp
        sL = DAY_END
    merged[-1] = ("home", sL, max(0, DAY_END - sL))

    # 7) sanity: no overlaps (exact integer checks)
    for i in range(1, len(merged)):
        prev_end = merged[i-1][1] + merged[i-1][2]
        if merged[i][1] < prev_end:
            # debug print to help if anything ever slips through
            print("DEBUG day segments:", merged)
            raise AssertionError(f"Overlap at idx {i}: {merged[i][1]} < prev_end {prev_end}")

    # 8) assert invariants in ticks (exact), then convert back to 3dp floats
    assert merged[0][0] == "home" and merged[0][1] == 0, "Day must start at 0 with 'home'"
    pL, sL, dL = merged[-1]
    assert pL == "home" and (sL + dL) == DAY_END, "Day must end at 24 with 'home'"

    out = [(p, round(_from_ticks(s_t), 3), round(_from_ticks(d_t), 3)) for (p, s_t, d_t) in merged]
    return out


def _write_purposes_csv(path: pathlib.Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["purpose","importance","flexibility","start_mu","start_std","dur_mu","dur_std","category","skip_prob"])
        for p in PURPOSES:
            feat = PURPOSE_FEATURES[p]
            w.writerow([p, feat["importance"], feat["flexibility"], feat["start_mu"], feat["start_std"],
                        feat["dur_mu"], feat["dur_std"], feat["category"], feat["skip_prob"]])

def generate_mock_data(out_dir="data/mock", n_persons=200, days_per_person=1, seed=DEFAULT_SEED):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    persons_path   = out / "persons.csv"
    schedules_path = out / "schedules.csv"
    purposes_path  = out / "purposes.csv"

    rng = np.random.default_rng(seed)

    # persons.csv
    with persons_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person_id","age","employment","household_size","num_children","car_ownership","home_zone"])
        for i in range(n_persons):
            pid = f"P{i:05d}"
            w.writerow(_person_row(rng, pid))

    # schedules.csv
    with schedules_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person_id","day","seq_id","purpose","start_time","duration"])
        for i in range(n_persons):
            pid = f"P{i:05d}"
            for d in range(days_per_person):
                day_label = f"weekday_{d}"
                segs = _sanitize_and_sort(_build_day_segments(rng))
                # inline asserts already enforce invariants; write rows
                for k, (p, s, dur) in enumerate(segs):
                    w.writerow([pid, day_label, k, p, f"{s:.3f}", f"{dur:.3f}"])

    # purposes.csv
    _write_purposes_csv(purposes_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/mock")
    parser.add_argument("--n_persons", type=int, default=200)
    parser.add_argument("--days_per_person", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    generate_mock_data(args.out_dir, args.n_persons, args.days_per_person, args.seed)
    print(f"Wrote {args.out_dir}/persons.csv, schedules.csv, purposes.csv")
