import torch
import pandas as pd


class ScheduleDataset(torch.utils.data.Dataset):
    """
    Reads activities and forms per-person sequences of (purpose, start_norm, dur_norm),
    normalized by the ALLOCATION horizon (e.g., 30h).
    """
    def __init__(self, activities_csv: str, T_alloc_minutes: int):
        acts = pd.read_csv(activities_csv)
        self.T = float(T_alloc_minutes)
        self.seqs = []
        for _, g in acts.groupby("persid"):
            g = g.sort_values(["startime", "stopno"])
            day = [
                (str(r["purpose"]), float(r["startime"] / self.T), float(r["total_duration"] / self.T))
                for _, r in g.iterrows()
            ]
            self.seqs.append(day)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx]


def collate_fn(batch, purpose_to_idx):
    p_lists, t_lists, d_lists, lens = [], [], [], []
    for seq in batch:
        p_idx = [purpose_to_idx[p] for p, _, _ in seq]
        t0 = [t for _, t, _ in seq]
        dd = [d for _, _, d in seq]
        p_lists.append(torch.tensor(p_idx, dtype=torch.long))
        t_lists.append(torch.tensor(t0, dtype=torch.float32))
        d_lists.append(torch.tensor(dd, dtype=torch.float32))
        lens.append(len(seq))
    p_pad = torch.nn.utils.rnn.pad_sequence(p_lists, batch_first=True, padding_value=0)
    t_pad = torch.nn.utils.rnn.pad_sequence(t_lists, batch_first=True, padding_value=0.0)
    d_pad = torch.nn.utils.rnn.pad_sequence(d_lists, batch_first=True, padding_value=0.0)
    return p_pad, t_pad, d_pad, lens
