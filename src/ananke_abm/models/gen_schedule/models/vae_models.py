import torch.nn as nn
import torch
from ananke_abm.models.gen_schedule.models.encoders import (
    ScheduleEncoderCNN,
    ScheduleEncoderRNN,
    reparameterize,
)
from ananke_abm.models.gen_schedule.models.decoders import ScheduleDecoderIndependent, ScheduleDecoderWithPDS


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


class ScheduleVAE_PDS(nn.Module):
    """
    VAE with:
      - CNN encoder (same as baseline)
      - PDS-conditioned decoder (ScheduleDecoderWithPDS)
    """

    def __init__(
        self,
        num_time_bins: int,
        num_purposes: int,
        z_dim: int,
        emb_dim: int,
        cnn_channels,
        cnn_kernel,
        cnn_dropout,
        pds_features: torch.Tensor,
    ):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.num_purposes = num_purposes
        self.z_dim = z_dim

        # encoder: same CNN you already use
        self.encoder = ScheduleEncoderCNN(
            P=num_purposes,
            T=num_time_bins,
            z_dim=z_dim,
            emb_dim=emb_dim,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            cnn_dropout=cnn_dropout,
        )

        # decoder: new PDS-aware decoder
        self.decoder = ScheduleDecoderWithPDS(
            num_time_bins=num_time_bins,
            num_purposes=num_purposes,
            z_dim=z_dim,
            emb_dim=emb_dim,
            pds_features=pds_features,
        )

    def encode(self, y_in_int):
        """
        y_in_int: [B,T] int64 labels.
        Returns mu, logvar each [B,z_dim]
        """
        return self.encoder(y_in_int)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y_in_int):
        """
        Full VAE step:
          - encode -> q(z|x)
          - sample z
          - decode -> logits
        Returns:
          logits [B,T,P],
          mu [B,z_dim],
          logvar [B,z_dim]
        """
        mu, logvar = self.encode(y_in_int)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample_from_prior(self, batch_size, device):
        """
        For generation:
          - sample z ~ N(0,I)
          - decode
        Returns logits [B,T,P]
        """
        z = torch.randn(batch_size, self.z_dim, device=device)
        logits = self.decoder(z)
        return logits
