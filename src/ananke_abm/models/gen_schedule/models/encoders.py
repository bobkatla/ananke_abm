import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ScheduleEncoderCNN(nn.Module):
    """
    Phase 1 encoder (what you have now), refactored into a module.

    y_seq: (B,T) long of purpose indices
    -> embedding -> 1D conv stack -> global average pool over time -> mu, logvar
    """

    def __init__(
        self,
        P,
        T,
        z_dim,
        emb_dim,
        cnn_channels=(64, 64),
        cnn_kernel=5,
        cnn_dropout=0.1,
    ):
        super().__init__()
        self.P = P
        self.T = T
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.embed = nn.Embedding(P, emb_dim)

        convs = []
        in_ch = emb_dim
        for ch in cnn_channels:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cnn_dropout),
                )
            )
            in_ch = ch
        self.conv_stack = nn.ModuleList(convs)

        self.mu_head = nn.Linear(in_ch, z_dim)
        self.logvar_head = nn.Linear(in_ch, z_dim)

    def forward(self, y_seq):
        """
        y_seq: (B,T) long
        returns (mu, logvar, z)
        """
        emb = self.embed(y_seq)          # (B,T,emb_dim)
        x = emb.transpose(1, 2)          # (B,emb_dim,T) for Conv1d

        for block in self.conv_stack:
            x = block(x)                 # (B,C,T)

        # global average pool over time
        x_pool = x.mean(dim=2)           # (B,C)

        mu = self.mu_head(x_pool)        # (B,z_dim)
        logvar = self.logvar_head(x_pool)# (B,z_dim)
        return mu, logvar


class ScheduleEncoderRNN(nn.Module):
    """
    New encoder for Phase 1.5:
    Bidirectional LSTM over embedded sequence.

    We summarize the whole sequence by concatenating the final
    forward hidden state and final backward hidden state.
    """

    def __init__(
        self,
        P,
        T,
        z_dim,
        emb_dim,
        rnn_hidden_dim=64,
        rnn_layers=1,
        rnn_dropout=0.1,
        use_emb_layernorm=False,
    ):
        super().__init__()
        self.P = P
        self.T = T
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.embed = nn.Embedding(P, emb_dim)
        self.use_emb_layernorm = use_emb_layernorm
        if use_emb_layernorm:
            self.emb_norm = nn.LayerNorm(emb_dim)

        # bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            dropout=(rnn_dropout if rnn_layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True,
        )
        # output dim after concat forward/backward last states
        enc_out_dim = 2 * rnn_hidden_dim

        self.mu_head = nn.Linear(enc_out_dim, z_dim)
        self.logvar_head = nn.Linear(enc_out_dim, z_dim)

    def forward(self, y_seq):
        """
        y_seq: (B,T) long
        returns (mu, logvar)
        """
        emb = self.embed(y_seq)      # (B,T,emb_dim)
        if self.use_emb_layernorm:
            emb = self.emb_norm(emb)

        # Run BiLSTM
        # rnn_out: (B,T,2*hidden_dim)
        # (h_n, c_n): each is (num_layers*2, B, hidden_dim)
        rnn_out, (h_n, c_n) = self.rnn(emb)

        # h_n layout:
        #   layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...
        # We want the last layer's forward and backward hidden states:
        # forward last = h_n[-2], backward last = h_n[-1]
        # but careful: if bidirectional=True, num_directions=2,
        # so last layer forward index = (rnn_layers-1)*2, backward = (rnn_layers-1)*2+1
        fwd_idx = (self.rnn.num_layers - 1) * 2 + 0
        bwd_idx = (self.rnn.num_layers - 1) * 2 + 1
        h_fwd_last = h_n[fwd_idx]    # (B, hidden_dim)
        h_bwd_last = h_n[bwd_idx]    # (B, hidden_dim)

        enc_summary = torch.cat([h_fwd_last, h_bwd_last], dim=-1)  # (B, 2*hidden_dim)

        mu = self.mu_head(enc_summary)         # (B,z_dim)
        logvar = self.logvar_head(enc_summary) # (B,z_dim)
        return mu, logvar
