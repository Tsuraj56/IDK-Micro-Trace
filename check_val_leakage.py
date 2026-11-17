import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from micro_trace.c1.models import run_c1_delta_benchmark

def main():
    cfg_path = Path("micro_trace/configs/c1_config.yaml")
    # Run only arm M1 with ElasticNetCV and MicroTrace-Attn-L1
    res = run_c1_delta_benchmark(
        cfg_path=cfg_path,
        arms=("M1",),
        families=("ElasticNetCV", "MicroTrace-Attn-L1"),
        epochs_attn=35,
        evaluate_test=False,
        output_csv_name="test_leakage_check.csv",
        output_json_name="test_leakage_check.json",
    )
    for row in res["summary_table"]:
        print(row)

if __name__ == "__main__":
    main()
