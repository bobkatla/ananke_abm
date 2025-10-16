import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ContRNNVAE(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int=256, rnn_hidden:int=256, rnn_layers:int=4,
                 latent_dim:int=6, dropout:float=0.1, max_len:int=20, teacher_forcing:float=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.tf = teacher_forcing

        # encoder
        self.emb_enc = nn.Embedding(vocab_size, emb_dim)
        self.enc_rnn = nn.LSTM(emb_dim+1, rnn_hidden, num_layers=rnn_layers,
                               batch_first=True, dropout=dropout)
        self.to_mu = nn.Linear(rnn_hidden, latent_dim)
        self.to_logvar = nn.Linear(rnn_hidden, latent_dim)

        # decoder
        self.emb_dec = nn.Embedding(vocab_size, emb_dim)
        self.z_to_h = nn.Linear(latent_dim, rnn_hidden*rnn_layers*2)  # h and c
        self.dec_rnn = nn.LSTM(emb_dim+1, rnn_hidden, num_layers=rnn_layers,
                               batch_first=True, dropout=dropout)
        self.head_act = nn.Linear(rnn_hidden, vocab_size)
        self.head_dur = nn.Linear(rnn_hidden, 1)

    def encode(self, acts, durs):
        # acts: (B,L) long, durs: (B,L) float
        x = torch.cat([self.emb_enc(acts), durs.unsqueeze(-1)], dim=-1)  # (B,L,emb+1)
        _, (h, _) = self.enc_rnn(x)  # h: (layers,B,H)
        h_last = h[-1]               # (B,H)
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_dec_state(self, z):
        B = z.size(0)
        hc = self.z_to_h(z)  # (B, 2*layers*H)
        H = self.rnn_hidden; L = self.rnn_layers
        hc = hc.view(B, 2, L, H).transpose(0,2)  # (layers, 2, B, H)
        h0 = hc[:,0]  # (layers,B,H)
        c0 = hc[:,1]
        return (h0.contiguous(), c0.contiguous())

    def decode(self, acts, durs, z, teacher_forcing: float = None):
        if teacher_forcing is None:
            teacher_forcing = self.tf
        B, L = acts.size()

        # init state from z
        h0, c0 = self.init_dec_state(z)

        # helper to build step input with correct dims
        def _step_input(act_ids, dur_scalar):
            # act_ids: (B,1), dur_scalar: (B,1) or (B,) -> return (B,1,E+1)
            emb = self.emb_dec(act_ids)                 # (B,1,E)
            if dur_scalar.dim() == 1:                   # (B,)
                dur3 = dur_scalar.view(B, 1, 1)         # (B,1,1)
            elif dur_scalar.dim() == 2:                 # (B,1)
                dur3 = dur_scalar.unsqueeze(-1)         # (B,1,1)
            else:
                # already (B,1,1)
                dur3 = dur_scalar
            return torch.cat([emb, dur3], dim=-1)       # (B,1,E+1)

        # first input is SOS with duration 0
        inputs = _step_input(acts[:, 0:1], durs[:, 0:1])  # SOS step
        state = (h0, c0)

        outputs_act = []
        outputs_dur = []

        for t in range(1, L):
            out, state = self.dec_rnn(inputs, state)    # out: (B,1,H)
            h_t = out[:, -1, :]                         # (B,H)

            logits_act = self.head_act(h_t)             # (B,V)
            logit_dur  = self.head_dur(h_t).squeeze(-1) # (B,)

            outputs_act.append(logits_act.unsqueeze(1)) # (B,1,V)
            outputs_dur.append(logit_dur.unsqueeze(1))  # (B,1)

            # next input (teacher forcing at sequence level — like the paper)
            use_tf = torch.rand((), device=acts.device) < teacher_forcing
            if use_tf:
                nxt_act = acts[:, t:t+1]                # (B,1)
                nxt_dur = durs[:, t:t+1]                # (B,1)
            else:
                nxt_act = torch.argmax(logits_act, dim=-1, keepdim=True)   # (B,1)
                nxt_dur = torch.sigmoid(logit_dur).unsqueeze(1)            # (B,1)

            inputs = _step_input(nxt_act, nxt_dur)

        logits_act = torch.cat(outputs_act, dim=1)      # (B, L-1, V)
        logits_dur = torch.cat(outputs_dur, dim=1)      # (B, L-1)
        return logits_act, logits_dur

    def forward(self, acts, durs):
        mu, logvar = self.encode(acts, durs)
        z = self.reparam(mu, logvar)
        logits_act, logits_dur = self.decode(acts, durs, z)
        return logits_act, logits_dur, mu, logvar

    @torch.no_grad()
    def sample(self, n, vocab, device, max_len=None):
        # greedy decode, then strip specials & renormalize durations to sum=1
        if max_len is None: max_len = self.max_len
        SOS = vocab["SOS"]; EOS = vocab["EOS"]
        z = torch.randn(n, self.latent_dim, device=device)
        h0, c0 = self.init_dec_state(z)
        prev_act = torch.full((n,1), SOS, dtype=torch.long, device=device)
        prev_dur = torch.zeros(n,1, device=device)
        inputs = torch.cat([self.emb_dec(prev_act), prev_dur], dim=-1)
        state = (h0, c0)

        acts_out = [prev_act]
        durs_out = [prev_dur]
        for t in range(1, max_len):
            out, state = self.dec_rnn(inputs, state)
            h_t = out[:, -1, :]
            logits_act = self.head_act(h_t)
            logit_dur = self.head_dur(h_t).squeeze(-1)
            next_act = torch.argmax(logits_act, dim=-1, keepdim=True)
            next_dur = torch.sigmoid(logit_dur).unsqueeze(-1)
            acts_out.append(next_act); durs_out.append(next_dur)
            inputs = torch.cat([self.emb_dec(next_act), next_dur], dim=-1)

        acts = torch.cat(acts_out, dim=1)      # (n,L)
        durs = torch.cat(durs_out, dim=1)      # (n,L)
        # strip SOS/EOS and renorm
        mask_special = (acts==SOS) | (acts==EOS)
        keep = ~mask_special
        # avoid empty rows: if all false, keep second token
        for i in range(n):
            if keep[i].sum()==0:
                keep[i,1]=True
        d_raw = durs * keep.float()
        s = d_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
        d_norm = d_raw / s
        acts_clean = acts[keep].view(n, -1)
        durs_clean = d_norm[keep].view(n, -1)
        return acts_clean, durs_clean
