import numpy as np
import torch
from ananke_abm.models.gen_schedule.models.vae_models import ScheduleVAE_CNNEnc, ScheduleVAE_RNNEnc, ScheduleVAE_PDS


def build_model(cfg, meta):
    """
    cfg: loaded YAML/JSON config dict
    meta: checkpoint/dataset meta (contains L and purpose_map)

    Expects:
    cfg["model"]["method"] in {"baseline_cnn", "baseline_rnn", "auto_pds"}
    """

    method = cfg["model"]["method"]
    num_time_bins = meta["L"]
    num_purposes = len(meta["purpose_map"])
    z_dim = cfg["model"]["z_dim"]
    emb_dim = cfg["model"]["emb_dim"]

    if method == "baseline_cnn":
        return ScheduleVAE_CNNEnc(
            L=num_time_bins,
            P=num_purposes,
            z_dim=z_dim,
            emb_dim=emb_dim,
            cnn_channels=cfg["model"].get("cnn_channels", [64, 64]),
            cnn_kernel=cfg["model"].get("cnn_kernel", 5),
            cnn_dropout=cfg["model"].get("cnn_dropout", 0.1),
        )

    elif method == "baseline_rnn":
        return ScheduleVAE_RNNEnc(
            L=num_time_bins,
            P=num_purposes,
            z_dim=z_dim,
            emb_dim=emb_dim,
            rnn_hidden_dim=cfg["model"].get("rnn_hidden_dim", 64),
            rnn_layers=cfg["model"].get("rnn_layers", 1),
            rnn_dropout=cfg["model"].get("rnn_dropout", 0.1),
            use_emb_layernorm=cfg["model"].get("use_emb_layernorm", False),
        )
    elif method == "auto_pds":
        # load PDS arrays from cfg["model"]["pds_path"]
        pds_npz = np.load(cfg["model"]["pds_path"])

        # We'll build a feature tensor phi_pds[p,t,D_pds].
        # Minimal features: m_tod and start_rate.
        m_tod = pds_npz["m_tod"].astype(np.float32)          # (P,T)
        start_rate = pds_npz["start_rate"].astype(np.float32) # (P,T)

        # stack along last dim -> (P,T,2)
        phi_pds = np.stack([m_tod, start_rate], axis=-1)      # (P,T,2)
        phi_pds_torch = torch.from_numpy(phi_pds)             # float32 tensor

        return ScheduleVAE_PDS(
            num_time_bins=num_time_bins,
            num_purposes=num_purposes,
            z_dim=cfg["model"]["z_dim"],
            emb_dim=cfg["model"]["emb_dim"],
            cnn_channels=cfg["model"]["cnn_channels"],
            cnn_kernel=cfg["model"]["cnn_kernel"],
            cnn_dropout=cfg["model"]["cnn_dropout"],
            pds_features=phi_pds_torch,
        )

    else:
        raise ValueError(f"Unknown model.method {method}")
