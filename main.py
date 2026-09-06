"""ENIA demo: one-day counterfactual EV-charging control.

Run with:

    python main.py

The script loads the rural low-voltage network, one daily EV/user profile, and
the pretrained Random Forest stability model.  It then simulates the same day
twice:

* uncontrolled baseline (all charging stations deliver full requested power);
* counterfactual control (the model is queried on several shading
  alternatives and the least-curtailing alternative predicted stable is
  selected, then verified by an AC power flow).

Results are saved in ``results/``.
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Silence third-party logging noise while preserving a concise demo log.
logging.getLogger("pypsa").setLevel(logging.WARNING)
logging.getLogger("pypsa.pf").setLevel(logging.WARNING)
logging.getLogger("pypsa.components").setLevel(logging.WARNING)

from controller import (
    find_controlled_shading,
    run_baseline_day,
    slice_ac_metrics,
    _is_stable,
)
from features import build_feature_vector
from network import Network


# ---------------------------------------------------------------------------
# Logging: one non-timestamped log file that is overwritten on every run.
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
LOGGER = logging.getLogger("enia_demo")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, "data", "simulation_data.json")
DAY_FILE = "Day_004_profiles.xlsx"
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_all_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_PATH = os.path.join(RESULTS_DIR, "one_day_control.png")
CSV_PATH = os.path.join(RESULTS_DIR, "one_day_metrics.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "one_day_summary.json")


def load_model_and_features() -> Tuple[object, List[str], float]:
    """Load the pretrained Random Forest and its expected feature order."""
    LOGGER.info("Loading pretrained model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["feature_names"]
    threshold = float(model_data.get("threshold", 0.5))

    LOGGER.info(
        "Model loaded: %s | features=%d | threshold=%.3f",
        type(model).__name__,
        len(feature_names),
        threshold,
    )
    return model, feature_names, threshold


def initialise_network() -> Network:
    """Load the network configuration, daily profiles, and AC_pf set-points."""
    LOGGER.info("Loading network configuration from %s", CFG_PATH)
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        param_dict = json.load(f)

    net = Network(param_dict, BASE_DIR, DAY_FILE)
    net.load_timeseries()
    net.update_with_optimization_results()

    LOGGER.info(
        "Network ready: %d buses, %d lines, %d available charging stations",
        len(net.bus_data),
        len(net.line_data),
        len(net.available_stations),
    )
    return net




def compute_summary_metrics(
    baseline_metrics: List[Dict], controlled_metrics: List[Dict]
) -> Dict:
    """Aggregate per-timestep metrics into daily summary values."""
    n = len(baseline_metrics)

    def stable_count(metrics_list):
        return sum(1 for m in metrics_list if m["ac_stable"])

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    base_delivered = [m["total_charging_kw"] for m in baseline_metrics]
    ctrl_delivered = [m["total_charging_kw"] for m in controlled_metrics]

    base_vmin = [m["min_voltage"] for m in baseline_metrics]
    ctrl_vmin = [m["min_voltage"] for m in controlled_metrics]
    base_vmax = [m["max_voltage"] for m in baseline_metrics]
    ctrl_vmax = [m["max_voltage"] for m in controlled_metrics]
    base_load = [m["max_loading"] for m in baseline_metrics]
    ctrl_load = [m["max_loading"] for m in controlled_metrics]

    return {
        "total_timesteps": n,
        "baseline_stable_timesteps": stable_count(baseline_metrics),
        "controlled_stable_timesteps": stable_count(controlled_metrics),
        "baseline_avg_charging_kw": avg(base_delivered),
        "controlled_avg_charging_kw": avg(ctrl_delivered),
        "curtailment_avg_kw": avg([b - c for b, c in zip(base_delivered, ctrl_delivered)]),
        "intervention_timesteps": [
            t
            for t, (b, c) in enumerate(zip(baseline_metrics, controlled_metrics))
            if b["shading_applied"] is False and c["shading_applied"] is True
        ],
        "baseline_min_voltage": min(base_vmin) if base_vmin else None,
        "controlled_min_voltage": min(ctrl_vmin) if ctrl_vmin else None,
        "baseline_max_loading": max(base_load) if base_load else None,
        "controlled_max_loading": max(ctrl_load) if ctrl_load else None,
    }


def extract_metrics(
    network: Network,
    timestep: int,
    shading: Dict[int, float],
    ac_metrics: Dict,
    ac_stable: bool,
    ml_score: float,
    shading_applied: bool,
) -> Dict:
    """Build a flat dictionary of per-timestep quantities for CSV/JSON export."""
    bus_names = [bus["name"] for bus in network.bus_data[2:]]
    line_names = [line["name"] for line in network.line_data[1:]]

    voltages = []
    if "Bus_V_Mag_PU" in ac_metrics:
        for name in bus_names:
            if name in ac_metrics["Bus_V_Mag_PU"].columns:
                voltages.append(float(ac_metrics["Bus_V_Mag_PU"][name].iloc[0]))

    loadings = []
    if "loading percentage" in ac_metrics:
        for name in line_names:
            if name in ac_metrics["loading percentage"].columns:
                loadings.append(float(ac_metrics["loading percentage"][name].iloc[0]))

    total_charging_kw = sum(
        network.get_charging_power_at_timestep(network.available_stations[i], timestep)
        * shading[i]
        for i in range(len(network.available_stations))
    )

    return {
        "timestep": timestep,
        "hour": timestep * network.tcf,
        "ac_stable": ac_stable,
        "ml_score": round(ml_score, 6),
        "min_voltage": round(min(voltages), 6) if voltages else None,
        "max_voltage": round(max(voltages), 6) if voltages else None,
        "max_loading": round(max(loadings), 6) if loadings else None,
        "total_charging_kw": round(total_charging_kw, 6),
        "avg_shading": round(np.mean(list(shading.values())), 6),
        "shading_applied": shading_applied,
    }


def run_one_day(
    network: Network, model, feature_names: List[str], threshold: float
) -> Tuple[List[Dict], List[Dict]]:
    """Simulate a full day for both the baseline and the controlled case."""
    baseline_records: List[Dict] = []
    controlled_records: List[Dict] = []

    LOGGER.info("Computing the 96-timestep uncontrolled baseline in one AC power flow...")
    baseline_network, baseline_day_metrics = run_baseline_day(network)
    full_power = {i: 1.0 for i in range(len(network.available_stations))}

    for timestep in range(network.n_ts):
        # --------------- baseline (no control) ---------------
        shading = full_power
        ac_metrics = slice_ac_metrics(baseline_day_metrics, timestep)
        ac_stable = _is_stable(network, ac_metrics)
        feature_vector = build_feature_vector(
            baseline_network, timestep, {"results_AC_ldf": ac_metrics}, shading, feature_names
        )
        baseline_score = float(model.predict(feature_vector.reshape(1, -1))[0])

        baseline_records.append(
            extract_metrics(
                network,
                timestep,
                shading,
                ac_metrics,
                ac_stable,
                baseline_score,
                shading_applied=False,
            )
        )

        # --------------- counterfactual control ---------------
        if ac_stable:
            controlled_shading = shading
            controlled_metrics = ac_metrics
            controlled_stable = True
            controlled_score = baseline_score
            shading_applied = False
        else:
            controlled_shading, info = find_controlled_shading(
                network, timestep, model, feature_names, threshold
            )
            controlled_metrics = info["metrics"]
            controlled_stable = info["ac_stable"]
            controlled_score = info["ml_score"]
            shading_applied = True

        controlled_records.append(
            extract_metrics(
                network,
                timestep,
                controlled_shading,
                controlled_metrics,
                controlled_stable,
                controlled_score,
                shading_applied,
            )
        )

        if (timestep + 1) % 12 == 0 or timestep == network.n_ts - 1:
            LOGGER.info(
                "Processed timestep %2d/%d  baseline_stable=%s  controlled_stable=%s  "
                "baseline_score=%.3f  controlled_score=%.3f",
                timestep + 1,
                network.n_ts,
                baseline_records[-1]["ac_stable"],
                controlled_records[-1]["ac_stable"],
                baseline_records[-1]["ml_score"],
                controlled_records[-1]["ml_score"],
            )

    return baseline_records, controlled_records


def plot_results(baseline: List[Dict], controlled: List[Dict]) -> None:
    """Create the comparison figure and save it to ``results/one_day_control.png``."""
    hours = [r["hour"] for r in baseline]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Stability / model score
    ax = axes[0]
    ax.plot(hours, [r["ml_score"] for r in baseline], label="Baseline", color="C3")
    ax.plot(hours, [r["ml_score"] for r in controlled], label="Controlled", color="C2")
    ax.axhline(0.8, color="k", linestyle="--", linewidth=1, label="Model threshold")
    ax.set_ylabel("RF stability score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.set_title("Day 004 – Counterfactual control mitigates network instability")

    # Min voltage
    ax = axes[1]
    ax.plot(hours, [r["min_voltage"] for r in baseline], label="Baseline", color="C3")
    ax.plot(hours, [r["min_voltage"] for r in controlled], label="Controlled", color="C2")
    ax.axhline(0.9, color="k", linestyle="--", linewidth=1, label="v_min")
    ax.set_ylabel("Minimum voltage [p.u.]")
    ax.legend(loc="best")

    # Max loading
    ax = axes[2]
    ax.plot(hours, [r["max_loading"] for r in baseline], label="Baseline", color="C3")
    ax.plot(hours, [r["max_loading"] for r in controlled], label="Controlled", color="C2")
    ax.axhline(100, color="k", linestyle="--", linewidth=1, label="load_max %")
    ax.set_ylabel("Maximum line loading [%]")
    ax.legend(loc="best")

    # Charging power
    ax = axes[3]
    ax.plot(hours, [r["total_charging_kw"] for r in baseline], label="Baseline", color="C3")
    ax.plot(hours, [r["total_charging_kw"] for r in controlled], label="Controlled", color="C2")
    ax.set_ylabel("Total charging power [kW]")
    ax.set_xlabel("Hour of day")
    ax.legend(loc="best")

    # Highlight intervention intervals
    intervention = [
        t
        for t, (b, c) in enumerate(zip(baseline, controlled))
        if not b["shading_applied"] and c["shading_applied"]
    ]
    for ax in axes:
        for t in intervention:
            ax.axvspan(hours[t] - 0.1, hours[t] + 0.1, color="yellow", alpha=0.2)

    ax = axes[0]
    if intervention:
        ax.axvspan(0, 0, color="yellow", alpha=0.2, label="Intervention")
        handles, labels = ax.get_legend_handles_labels()
        # Keep only unique labels.
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    LOGGER.info("Plot saved to %s", PLOT_PATH)


def save_results(baseline: List[Dict], controlled: List[Dict], summary: Dict) -> None:
    """Write CSV and JSON summaries."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_base = pd.DataFrame(baseline)
    df_ctrl = pd.DataFrame(controlled)
    df = df_base.merge(
        df_ctrl,
        on="timestep",
        suffixes=("_baseline", "_controlled"),
    )
    df.to_csv(CSV_PATH, index=False)
    LOGGER.info("Metrics CSV saved to %s", CSV_PATH)

    summary["generated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Summary JSON saved to %s", JSON_PATH)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="ENIA demo: one-day counterfactual EV-charging control."
    )
    parser.add_argument(
        "--day",
        default="Day_004_profiles.xlsx",
        help="Daily profile workbook under data/ (default: Day_004_profiles.xlsx).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    global DAY_FILE
    DAY_FILE = args.day

    LOGGER.info("=" * 60)
    LOGGER.info("ENIA award demo – one-day counterfactual control")
    LOGGER.info("Day file: %s", DAY_FILE)
    LOGGER.info("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, feature_names, threshold = load_model_and_features()
    network = initialise_network()

    LOGGER.info("Running baseline and controlled simulations for one day...")
    baseline, controlled = run_one_day(network, model, feature_names, threshold)

    summary = compute_summary_metrics(baseline, controlled)

    LOGGER.info("Simulation complete.")
    LOGGER.info(
        "Baseline stable timesteps: %d/%d",
        summary["baseline_stable_timesteps"],
        summary["total_timesteps"],
    )
    LOGGER.info(
        "Controlled stable timesteps: %d/%d",
        summary["controlled_stable_timesteps"],
        summary["total_timesteps"],
    )
    LOGGER.info(
        "Intervention timesteps: %s",
        summary["intervention_timesteps"],
    )
    LOGGER.info(
        "Average charging power: baseline %.2f kW, controlled %.2f kW",
        summary["baseline_avg_charging_kw"],
        summary["controlled_avg_charging_kw"],
    )

    plot_results(baseline, controlled)
    save_results(baseline, controlled, summary)

    LOGGER.info("Demo finished successfully. Results are in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
