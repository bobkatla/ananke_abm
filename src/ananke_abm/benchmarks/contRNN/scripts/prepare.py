import yaml
from ananke_abm.benchmarks.contRNN.data_process.prepare_from_csv import prepare_from_csv

cfg = yaml.safe_load(open("src/ananke_abm/benchmarks/contRNN/configs/cont_rnn.yaml"))
d = cfg["data"]
info = prepare_from_csv(
    csv_path=d["csv_path"],
    out_dir=d["out_dir"],
    max_len=d["max_len"],
    day_minutes=d["day_minutes"],
    splits=tuple(d["splits"])
)
print("Saved:", info["paths"])
