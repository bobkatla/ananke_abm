from ananke_abm.models.gen_schedule.compare.extract_metrics.general import GENERAL_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.raw_counts import RAW_COUNTS_FUNCS
    
METRIC_FUNCS = {
    **GENERAL_FUNCS,
    **RAW_COUNTS_FUNCS,
}
