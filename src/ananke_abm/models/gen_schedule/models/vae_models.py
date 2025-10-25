import torch.nn as nn
from ananke_abm.models.gen_schedule.models.encoders import (
    ScheduleEncoderCNN,
    ScheduleEncoderRNN,
    reparameterize,
)
from ananke_abm.models.gen_schedule.models.decoders import ScheduleDecoderIndependent


class ScheduleVAE_CNNEnc(nn.Module):
    """
    VAE with CNN encoder + independent-per-time decoder.
    API-compatible with old ScheduleVAE.
    """

    def __init__(
        self,
        L,
        P,
        z_dim,
        emb_dim,
        cnn_channels=(64, 64),
        cnn_kernel=5,
        cnn_dropout=0.1,
    ):
        super().__init__()
        self.L = L
        self.P = P
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.encoder = ScheduleEncoderCNN(
            P=P,
            T=L,
            z_dim=z_dim,
            emb_dim=emb_dim,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            cnn_dropout=cnn_dropout,
        )

        self.decoder = ScheduleDecoderIndependent(
            L=L,
            P=P,
            z_dim=z_dim,
            emb_dim=emb_dim,
        )

    def forward(self, y_seq):
        """
        y_seq: (B,T) long
        returns logits: (B,T,P), mu: (B,z_dim), logvar: (B,z_dim)
        """
        mu, logvar = self.encoder(y_seq)
        z = reparameterize(mu, logvar)    # (B,z_dim)
        logits = self.decoder(z)          # (B,T,P)
        return logits, mu, logvar


class ScheduleVAE_RNNEnc(nn.Module):
    """
    VAE with BiLSTM encoder + same decoder.
    """

    def __init__(
        self,
        L,
        P,
        z_dim,
        emb_dim,
        rnn_hidden_dim=64,
        rnn_layers=1,
        rnn_dropout=0.1,
        use_emb_layernorm=False,
    ):
        super().__init__()
        self.L = L
        self.P = P
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.encoder = ScheduleEncoderRNN(
            P=P,
            T=L,
            z_dim=z_dim,
            emb_dim=emb_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            use_emb_layernorm=use_emb_layernorm,
        )

        self.decoder = ScheduleDecoderIndependent(
            L=L,
            P=P,
            z_dim=z_dim,
            emb_dim=emb_dim,
        )

    def forward(self, y_seq):
        """
        y_seq: (B,T) long
        returns logits: (B,T,P), mu: (B,z_dim), logvar: (B,z_dim)
        """
        mu, logvar = self.encoder(y_seq)
        z = reparameterize(mu, logvar)   # (B,z_dim)
        logits = self.decoder(z)         # (B,T,P)
        return logits, mu, logvar
