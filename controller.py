"""Counterfactual control logic.

At every timestep the demo first runs the uncontrolled case (all charging
stations deliver full power).  If the AC power flow reports instability, a
greedy search over per-station shading factors finds the smallest curtailment
that restores stability.  The selected assignment is then re-evaluated with the
pretrained Random Forest so the final report contains both the physical AC
stability flag and the model's stability score.
"""

import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ac_loadflow import run_ac_load_flow
from features import build_feature_vector
from network import Network


def evaluate_shading(
    network: Network,
    timestep: int,
    shading: Dict[int, float],
    model,
    feature_names: List[str],
) -> Tuple[bool, float, Dict]:
    """Apply a shading candidate, run AC power flow, and score it with the model.

    Returns
    -------
    is_ac_stable: True if the AC power flow is within voltage/loading limits.
    ml_score: model prediction for stability (higher is more stable).
    info: dictionary with AC results and the feature vector used for scoring.
    """
    net = _apply_shading_to_network(network, timestep, shading)
    metrics = _run_ac_for_timestep(net, timestep)
    is_ac_stable = _is_stable(net, metrics)

    feature_vector = build_feature_vector(
        net, timestep, {"results_AC_ldf": metrics}, shading, feature_names
    )
    ml_score = float(model.predict(feature_vector.reshape(1, -1))[0])

    return is_ac_stable, ml_score, {"metrics": metrics, "ac_stable": is_ac_stable, "ml_score": ml_score}


def run_baseline_day(network: Network) -> Tuple[Network, Dict]:
    """Run the full-power baseline for all 96 timesteps in one AC power flow."""
    net = copy.deepcopy(network)
    tanfi = math.tan(math.acos(0.99))

    for station_id in net.available_stations:
        bus_idx = net.station_to_bus(station_id)
        if bus_idx is None:
            continue
        charging_mw = pd.Series(
            [
                net.get_charging_power_at_timestep(station_id, t) / 1000.0
                for t in range(net.n_ts)
            ]
        )
        for load in net.load_data:
            if load.get("bus_index") == bus_idx:
                load["p_mw"] = load["p_mw"].reset_index(drop=True) + charging_mw
                load["q_mvar"] = load["q_mvar"].reset_index(drop=True) + charging_mw * tanfi
                break

    metrics = run_ac_load_flow(
        net.bus_data,
        net.load_data,
        net.sgen_data,
        net.line_data,
        net.ext_grid_data,
        net.n_ts,
        net.tcf,
        net.transformer_data,
    )
    return net, metrics


def slice_ac_metrics(metrics: Dict, timestep: int) -> Dict:
    """Extract one timestep from a multi-snapshot AC result dictionary."""
    sliced = {}
    for key, value in metrics.items():
        if isinstance(value, (pd.DataFrame, pd.Series)) and len(value.index) > timestep:
            sliced[key] = value.iloc[[timestep]].reset_index(drop=True)
        else:
            sliced[key] = value
    return sliced


def run_baseline(
    network: Network, timestep: int, model=None, feature_names=None
) -> Tuple[Dict, Dict, bool, float]:
    """Run the uncontrolled case (full power to every station) at one timestep.

    If ``model`` and ``feature_names`` are supplied, the baseline Random Forest
    stability score is also returned; otherwise the score is NaN.
    """
    n_stations = len(network.available_stations)
    shading = {i: 1.0 for i in range(n_stations)}
    net_with_charging = _apply_shading_to_network(network, timestep, shading)
    metrics = _run_ac_for_timestep(net_with_charging, timestep)
    stable = _is_stable(net_with_charging, metrics)

    score = float("nan")
    if model is not None and feature_names is not None:
        feature_vector = build_feature_vector(
            net_with_charging, timestep, {"results_AC_ldf": metrics}, shading, feature_names
        )
        score = float(model.predict(feature_vector.reshape(1, -1))[0])

    return shading, metrics, stable, score


def find_controlled_shading(
    network: Network,
    timestep: int,
    model,
    feature_names: List[str],
    threshold: float,
) -> Tuple[Dict[int, float], Dict]:
    """Find a low-curtailment shading assignment that restores AC stability.

    A small set of counterfactual shading assignments is evaluated with the AC
    power flow:

    * full power (baseline, known unstable here);
    * each station turned off individually;
    * all stations turned off.

    The highest-power assignment that is both predicted stable by the Random
    Forest and verified stable by AC power flow is selected. This is a compact
    discrete counterfactual search suitable for a self-contained demo.
    """
    n_stations = len(network.available_stations)

    # Build candidate list.  To keep the demo fast we test a small, interpretable
    # set of shading assignments with real AC power flows:
    #   * the full-power baseline (known unstable here);
    #   * the three highest-power single-station curtailments (0 % for one station);
    #   * uniform global curtailments (75 %, 50 %, 25 % of nominal for all);
    #   * full curtailment (0 % for all).
    # The highest-power AC-stable candidate is selected.
    candidates = []

    # All-on (baseline) – included for completeness, although the caller only
    # invokes this function when the baseline is already unstable.
    candidates.append({i: 1.0 for i in range(n_stations)})

    # Single-station curtailment for the top three active stations.
    powers = [
        (i, network.get_charging_power_at_timestep(network.available_stations[i], timestep))
        for i in range(n_stations)
    ]
    for i, _ in sorted(powers, key=lambda x: x[1], reverse=True)[:3]:
        shading = {j: 1.0 for j in range(n_stations)}
        shading[i] = 0.0
        candidates.append(shading)

    # Uniform global curtailment levels.
    for level in (0.75, 0.5, 0.25):
        candidates.append({i: level for i in range(n_stations)})

    # Full curtailment (guaranteed stable because user load alone is stable).
    candidates.append({i: 0.0 for i in range(n_stations)})

    best_shading = candidates[-1]
    best_power = 0.0
    best_info = None

    for shading in candidates:
        is_ac_stable, ml_score, info = evaluate_shading(
            network, timestep, shading, model, feature_names
        )
        delivered = sum(
            network.get_charging_power_at_timestep(network.available_stations[i], timestep)
            * shading[i]
            for i in range(n_stations)
        )
        if is_ac_stable and ml_score >= threshold and delivered > best_power:
            best_power = delivered
            best_shading = shading
            best_info = info

    if best_info is None:
        # Should not happen because the all-off candidate is always evaluated.
        _, _, best_info = evaluate_shading(
            network, timestep, best_shading, model, feature_names
        )

    best_info["predicted_stable"] = best_info["ml_score"] >= threshold
    return best_shading, best_info


def _run_ac_for_timestep(network: Network, timestep: int) -> Dict:
    """Run an AC power flow for one timestep using the current network state."""
    single_ts_load_data = []
    for load in network.load_data:
        single_load = load.copy()
        p_series = load.get("p_mw")
        q_series = load.get("q_mvar")
        if isinstance(p_series, pd.Series) and timestep < p_series.shape[0]:
            single_load["p_mw"] = pd.Series([p_series.iloc[timestep]])
            single_load["q_mvar"] = pd.Series([q_series.iloc[timestep]])
        else:
            single_load["p_mw"] = pd.Series([0.0])
            single_load["q_mvar"] = pd.Series([0.0])
        single_ts_load_data.append(single_load)

    single_ts_sgen_data = []
    for sgen in network.sgen_data:
        single_sgen = sgen.copy()
        p_series = sgen.get("p_mw")
        q_series = sgen.get("q_mvar")
        if isinstance(p_series, pd.Series) and timestep < p_series.shape[0]:
            single_sgen["p_mw"] = pd.Series([p_series.iloc[timestep]])
            single_sgen["q_mvar"] = pd.Series([q_series.iloc[timestep]])
        else:
            single_sgen["p_mw"] = pd.Series([0.0])
            single_sgen["q_mvar"] = pd.Series([0.0])
        single_ts_sgen_data.append(single_sgen)

    return run_ac_load_flow(
        network.bus_data,
        single_ts_load_data,
        single_ts_sgen_data,
        network.line_data,
        network.ext_grid_data,
        1,
        network.tcf,
        network.transformer_data,
    )


def _apply_shading_to_network(
    network: Network, timestep: int, shading: Dict[int, float]
) -> Network:
    """Return a deep copy of ``network`` with the shaded EV charging loads added."""
    net = copy.deepcopy(network)
    tanfi = math.tan(math.acos(0.99))

    for station_idx, station_id in enumerate(net.available_stations):
        bus_idx = net.station_to_bus(station_id)
        if bus_idx is None:
            continue

        power_kw = net.get_charging_power_at_timestep(station_id, timestep)
        power_mw = power_kw / 1000.0
        actual_power = power_mw * shading.get(station_idx, 1.0)

        for load in net.load_data:
            if load.get("bus_index") == bus_idx:
                load["p_mw"].iloc[timestep] += actual_power
                load["q_mvar"].iloc[timestep] += actual_power * tanfi
                break

    return net


def _is_stable(network: Network, metrics: Dict) -> bool:
    """Check voltage and loading limits used by the project."""
    margin = _stability_margin(network, metrics)
    return margin >= 0.0


def _stability_margin(network: Network, metrics: Dict) -> float:
    """Return the minimum distance to any instability threshold.

    A positive value means all voltages and loadings are within limits.  The
    most violated quantity drives the greedy search.
    """
    bus_columns = [bus["name"] for bus in network.bus_data[2:]]
    line_columns = [line["name"] for line in network.line_data[1:]]

    margins = []

    if "Bus_V_Mag_PU" in metrics:
        for bus_name in bus_columns:
            if bus_name in metrics["Bus_V_Mag_PU"].columns:
                v = float(metrics["Bus_V_Mag_PU"][bus_name].iloc[0])
                margins.append(v - network.v_min)  # undervoltage margin
                margins.append(network.v_max - v)  # overvoltage margin

    if "loading percentage" in metrics:
        for line_name in line_columns:
            if line_name in metrics["loading percentage"].columns:
                loading = float(metrics["loading percentage"][line_name].iloc[0])
                margins.append(network.load_max - loading)

    return min(margins) if margins else -np.inf
