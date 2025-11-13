from ananke_abm.models.gen_schedule.compare.extract_metrics.general import GENERAL_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.raw_counts import RAW_COUNTS_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.srmse import SRMSE_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.tod_jsd import TOD_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.duration_jsd import DURATION_FUNCS
from ananke_abm.models.gen_schedule.compare.extract_metrics.diversity import DIVERSITY_FUNCS
    
METRIC_FUNCS = {
    **TOD_FUNCS,
    **DURATION_FUNCS,
    **GENERAL_FUNCS,
    **RAW_COUNTS_FUNCS,
    **SRMSE_FUNCS,
    **DIVERSITY_FUNCS,
}
