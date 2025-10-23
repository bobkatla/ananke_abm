import torch
import torch.nn as nn

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
        H = self.rnn_hidden
        L = self.rnn_layers
        hc = hc.view(B, 2, L, H).transpose(0,2)  # (layers, 2, B, H)
        h0 = hc[:,0]  # (layers,B,H)
        c0 = hc[:,1]
        return (h0.contiguous(), c0.contiguous())

    def decode(self, acts, durs, z, teacher_forcing: float = None):
        """
        Paper-faithful decode for training:
        - Per-sample TF with prob=teacher_forcing
        - Do NOT teacher-force beyond the first EOS
        - Forbid predicting specials (SOS/EOS) as next token
        - Build inputs with consistent shape (B,1,E+1)
        """
        if teacher_forcing is None:
            teacher_forcing = self.tf
        B, L = acts.size()
        device = acts.device
        SOS, EOS = 0, 1  # by our vocab construction

        # compute TF horizon (number of non-special targets + SOS step)
        with torch.no_grad():
            target_mask = ((acts[:, 1:] != SOS) & (acts[:, 1:] != EOS))  # (B, L-1)
            len_tf = 1 + target_mask.sum(dim=1)                           # (B,)

        def _step_input(act_ids, dur_scalar):
            emb = self.emb_dec(act_ids)                 # (B,1,E)
            if dur_scalar.dim() == 1:                   # (B,)
                dur3 = dur_scalar.view(B,1,1)
            elif dur_scalar.dim() == 2:                 # (B,1)
                dur3 = dur_scalar.unsqueeze(-1)         # (B,1,1)
            else:
                dur3 = dur_scalar                       # (B,1,1)
            return torch.cat([emb, dur3], dim=-1)       # (B,1,E+1)

        # init state from z
        h0, c0 = self.init_dec_state(z)
        state = (h0, c0)

        # first input = SOS, duration=0
        cur_act = acts[:, 0:1]                          # (B,1)
        cur_dur = durs[:, 0:1]                          # (B,1)
        inputs  = _step_input(cur_act, cur_dur)

        out_act, out_dur = [], []

        for t in range(1, L):
            out, state = self.dec_rnn(inputs, state)    # (B,1,H)
            h_t = out[:, -1, :]                         # (B,H)

            logits_act = self.head_act(h_t)             # (B,V)
            logit_dur  = self.head_dur(h_t).squeeze(-1) # (B,)

            # forbid predicting specials
            logits_act[:, [SOS, EOS]] = float("-inf")

            out_act.append(logits_act.unsqueeze(1))     # (B,1,V)
            out_dur.append(logit_dur.unsqueeze(1))      # (B,1)

            # per-sample TF, but not past first EOS
            tf_sample = (torch.rand(B, 1, device=device) < teacher_forcing)
            before_eos = (t < len_tf).view(B,1)
            use_tf = tf_sample & before_eos

            pred_act = torch.argmax(logits_act, dim=-1, keepdim=True)  # (B,1)
            pred_dur = torch.sigmoid(logit_dur).unsqueeze(1)           # (B,1)

            next_act = torch.where(use_tf, acts[:, t:t+1], pred_act)
            next_dur = torch.where(use_tf, durs[:, t:t+1], pred_dur)

            inputs = _step_input(next_act, next_dur)

        logits_act = torch.cat(out_act, dim=1)          # (B, L-1, V)
        logits_dur = torch.cat(out_dur, dim=1)          # (B, L-1)
        return logits_act, logits_dur

    @torch.no_grad()
    def sample(self, n, vocab, device, max_len=None):
        """
        Greedy decode (deterministic argmax) for inference:
        - start with SOS
        - forbid specials during argmax
        - strip specials and renormalise durations to sum=1
        """
        if max_len is None:
            max_len = self.max_len
        SOS = vocab["SOS"]; EOS = vocab["EOS"]

        def _step_input(act_ids, dur_scalar):
            emb = self.emb_dec(act_ids)                 # (B,1,E)
            if dur_scalar.dim() == 1:                   # (B,)
                dur3 = dur_scalar.view(-1,1,1)
            elif dur_scalar.dim() == 2:                 # (B,1)
                dur3 = dur_scalar.unsqueeze(-1)         # (B,1,1)
            else:
                dur3 = dur_scalar
            return torch.cat([emb, dur3], dim=-1)

        z = torch.randn(n, self.latent_dim, device=device)
        h0, c0 = self.init_dec_state(z)
        state = (h0, c0)

        prev_act = torch.full((n,1), SOS, dtype=torch.long, device=device)
        prev_dur = torch.zeros(n,1, device=device)
        inputs   = _step_input(prev_act, prev_dur)

        acts_steps = [prev_act]
        durs_steps = [prev_dur]

        for t in range(1, max_len):
            out, state = self.dec_rnn(inputs, state)
            h_t = out[:, -1, :]
            logits_act = self.head_act(h_t)
            # forbid specials
            logits_act[:, [SOS, EOS]] = float("-inf")
            logit_dur = self.head_dur(h_t).squeeze(-1)

            next_act = torch.argmax(logits_act, dim=-1, keepdim=True)  # (B,1)
            next_dur = torch.sigmoid(logit_dur).unsqueeze(1)           # (B,1)

            acts_steps.append(next_act)
            durs_steps.append(next_dur)

            inputs = _step_input(next_act, next_dur)

        acts = torch.cat(acts_steps, dim=1)       # (n,L)
        durs = torch.cat(durs_steps, dim=1)       # (n,L)

        # strip specials and renormalise
        keep = ~((acts == SOS) | (acts == EOS))
        # avoid empty rows
        for i in range(n):
            if keep[i].sum() == 0:
                keep[i, 1] = True
        d_raw = durs * keep.float()
        s = d_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
        d_norm = d_raw / s

        acts_clean = acts[keep].view(n, -1)
        durs_clean = d_norm[keep].view(n, -1)
        return acts_clean, durs_clean

    def forward(self, acts, durs):
        mu, logvar = self.encode(acts, durs)
        z = self.reparam(mu, logvar)
        logits_act, logits_dur = self.decode(acts, durs, z)
        return logits_act, logits_dur, mu, logvar