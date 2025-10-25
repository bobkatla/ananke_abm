from ananke_abm.models.gen_schedule.models.vae_models import ScheduleVAE_CNNEnc, ScheduleVAE_RNNEnc


def build_model(cfg, meta):
    """
    cfg: loaded YAML/JSON config dict
    meta: checkpoint/dataset meta (contains L and purpose_map)

    Expects:
    cfg["model"]["method"] in {"baseline_cnn", "baseline_rnn"}
    """

    method = cfg["model"]["method"]
    L = meta["L"]
    P = len(meta["purpose_map"])
    z_dim = cfg["model"]["z_dim"]
    emb_dim = cfg["model"]["emb_dim"]

    if method == "baseline_cnn":
        return ScheduleVAE_CNNEnc(
            L=L,
            P=P,
            z_dim=z_dim,
            emb_dim=emb_dim,
            cnn_channels=cfg["model"].get("cnn_channels", [64, 64]),
            cnn_kernel=cfg["model"].get("cnn_kernel", 5),
            cnn_dropout=cfg["model"].get("cnn_dropout", 0.1),
        )

    elif method == "baseline_rnn":
        return ScheduleVAE_RNNEnc(
            L=L,
            P=P,
            z_dim=z_dim,
            emb_dim=emb_dim,
            rnn_hidden_dim=cfg["model"].get("rnn_hidden_dim", 64),
            rnn_layers=cfg["model"].get("rnn_layers", 1),
            rnn_dropout=cfg["model"].get("rnn_dropout", 0.1),
            use_emb_layernorm=cfg["model"].get("use_emb_layernorm", False),
        )

    else:
        raise ValueError(f"Unknown model.method {method}")
