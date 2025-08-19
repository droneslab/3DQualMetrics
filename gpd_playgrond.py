# %% Artifact intensity (from decimal-less filenames) vs measured average artifacts

import os, re, json, glob
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== CONFIG ======
RESULTS_DIR = "./results/round_0"  # <- change to your folder with the result JSONs
PATTERN = "artifact_*after*.json"       # only artifact result files

# ====== HELPERS ======
def to_decimal_from_zeros(token: str) -> float:
    """
    Convert tokens like '01','0001','005','0005','1','10' to decimals
    by placing a decimal before the LAST digit:
      '01'   -> 0.1
      '0001' -> 0.001
      '005'  -> 0.05
      '0005' -> 0.005
      '1'    -> 1.0
      '10'   -> 1.0  (10 / 10^(2-1))
    If token already has a '.', just float() it.
    """
    token = token.strip()
    if "." in token:
        return float(token)
    if not token.isdigit():
        raise ValueError(f"Non-numeric token: {token}")
    n = len(token)
    return int(token) / (10 ** (n - 1)) if n > 1 else float(token)

def intensity_from_result_filename(fname: str) -> float:
    """
    Expect names like:
      artifact_01_002_0_after_size_...json
      artifact_0005_002_0_before_size_...json
    Grab the first numeric token after 'artifact_' and convert it.
    """
    base = os.path.basename(fname)
    if not base.lower().startswith("artifact_"):
        return None
    parts = base.split("_")
    if len(parts) < 2:
        return None
    return to_decimal_from_zeros(parts[1])

def get_measured_artifacts(payload: dict) -> float:
    """Pull measured average artifacts from common locations."""
    avg = payload.get("average")
    if isinstance(avg, dict) and "artifacts" in avg:
        return float(avg["artifacts"])
    metrics = payload.get("metrics")
    # if isinstance(metrics, dict):
    #     vals = [v.get("artifacts") for v in metrics.values()
    #             if isinstance(v, dict) and "artifacts" in v]
    #     if vals:
    #         return float(mean(vals))
    return None

# ====== MAIN ======
rows = []
for path in glob.glob(os.path.join(RESULTS_DIR, PATTERN)):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        continue

    measured = get_measured_artifacts(data)
    if measured is None:
        continue

    intensity = intensity_from_result_filename(path)
    if intensity is None:
        continue

    rows.append({
        "result_json": os.path.basename(path),
        "artifact_intensity": intensity,
        "measured_artifacts": measured,
    })

df = pd.DataFrame(rows).sort_values("artifact_intensity").reset_index(drop=True)
print(df)


if not df.empty:
    plt.figure()
    plt.scatter(df["artifact_intensity"], df["measured_artifacts"])
    plt.xlabel("Artifact intensity (decoded from filename zeros)")
    plt.ylabel("Measured average artifacts (from JSON)")
    plt.title("Artifacts: Intended Intensity vs Measured Average")
    plt.grid(True)
    plt.show()
else:
    print("No rows parsed. Check RESULTS_DIR and filename pattern.")