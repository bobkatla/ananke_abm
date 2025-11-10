from typing import Dict, List


# --------------------------------------------------------------------------
# Metric stubs (phase 0: not implemented yet)
# --------------------------------------------------------------------------

def metric_minutes_share_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'minutes_share' not implemented yet")


def metric_tod_jsd_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'tod_jsd' not implemented yet")


def metric_bigram_L1_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'bigram_L1' not implemented yet")


def metric_structure_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'structure' not implemented yet")


def metric_schedule_feasibility_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'schedule_feasibility' not implemented yet")


def metric_ngram_stub(ref: Dict, models: List[Dict], outdir: str):
    raise NotImplementedError("metric 'ngram' not implemented yet")


METRIC_FUNCS = {
    "minutes_share":        metric_minutes_share_stub,
    "tod_jsd":              metric_tod_jsd_stub,
    "bigram_L1":            metric_bigram_L1_stub,
    "structure":            metric_structure_stub,
    "schedule_feasibility": metric_schedule_feasibility_stub,
    "ngram":                metric_ngram_stub,
}
