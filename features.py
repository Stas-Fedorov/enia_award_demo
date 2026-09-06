"""Feature extraction for the pretrained Random Forest stability model.

The pretrained model expects exactly 127 features in a fixed order.  This
module builds that vector from the network state, the AC power-flow results,
and a candidate set of shading factors.
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from network import Network


def build_feature_vector(
    network: Network,
    timestep: int,
    stability_metrics: Dict,
    shading_factors: Dict[int, float],
    feature_names: List[str],
) -> np.ndarray:
    """Build a feature array matching the order expected by the classifier.

    Parameters
    ----------
    network: loaded Network instance.
    timestep: integer index in [0, n_ts).
    stability_metrics: dictionary containing the AC load-flow result under
        the candidate shading (key ``results_AC_ldf``).
    shading_factors: mapping station_index -> shading factor in [0, 1].
    feature_names: ordered list of feature names saved with the model.

    Returns
    -------
    1-D numpy array ready for ``model.predict``.
    """
    features = _extract_features(network, timestep, stability_metrics, shading_factors)
    # Reorder to the model's expected column order. In the three-station demo,
    # unused station controls remain at full power in the model input.
    vector = [
        features.get(name, 1.0 if name.startswith("station_") else 0.0)
        for name in feature_names
    ]
    return np.array(vector, dtype=float)


def _extract_features(
    network: Network,
    timestep: int,
    stability_metrics: Dict,
    shading_factors: Dict[int, float],
) -> Dict[str, float]:
    """Return a dictionary with all 127 features.

    Feature groups follow the order used during training:
    - station shading factors
    - bus voltage magnitudes
    - line loading percentages
    - per-line active power losses
    - load active/reactive powers per user
    - generator active/reactive powers per user
    """
    features: Dict[str, float] = {}

    # Station shading factors for the available stations.  Stations are indexed
    # 0..N-1 in the order Network.available_stations returns.
    available = network.available_stations
    for idx, _ in enumerate(available):
        features[f"station_{idx}_shading_factor"] = float(
            shading_factors.get(idx, 1.0)
        )

    results = stability_metrics.get("results_AC_ldf", {})

    # Bus voltages
    if "Bus_V_Mag_PU" in results:
        bus_voltages = results["Bus_V_Mag_PU"]
        if isinstance(bus_voltages, pd.DataFrame):
            for col in bus_voltages.columns:
                value = bus_voltages.iloc[0][col]
                features[f"voltage_{col}"] = float(value) if pd.notna(value) else 0.0

    # Line loading
    if "loading percentage" in results:
        line_loading = results["loading percentage"]
        if isinstance(line_loading, pd.DataFrame):
            for col in line_loading.columns:
                value = line_loading.iloc[0][col]
                features[f"loading_{col}"] = float(value) if pd.notna(value) else 0.0

    # Power losses
    if "p_losses" in results:
        p_losses = results["p_losses"]
        if hasattr(p_losses, "iloc"):
            p_losses_series = p_losses.iloc[0]
            if isinstance(p_losses_series, pd.Series):
                for line_name, loss_value in p_losses_series.items():
                    features[f"p_losses_{line_name}"] = (
                        float(loss_value) if pd.notna(loss_value) else 0.0
                    )

    # Load and generation profiles
    tanfi = math.tan(math.acos(0.99))
    for i, load in enumerate(network.load_data):
        user_num = i + 1
        if not load.get("empty", False) and isinstance(load.get("p_mw"), pd.Series):
            p_value = (
                load["p_mw"].iloc[timestep]
                if timestep < load["p_mw"].shape[0]
                else 0.0
            )
            q_value = (
                load["q_mvar"].iloc[timestep]
                if timestep < load["q_mvar"].shape[0]
                else 0.0
            )
            features[f"load_user_{user_num}_p"] = float(p_value)
            features[f"load_user_{user_num}_q"] = float(q_value)

    for i, sgen in enumerate(network.sgen_data):
        user_num = i + 1
        if not sgen.get("empty", False) and isinstance(sgen.get("p_mw"), pd.Series):
            p_value = (
                sgen["p_mw"].iloc[timestep]
                if timestep < sgen["p_mw"].shape[0]
                else 0.0
            )
            q_value = (
                sgen["q_mvar"].iloc[timestep]
                if timestep < sgen["q_mvar"].shape[0]
                else 0.0
            )
            features[f"gen_user_{user_num}_p"] = float(p_value)
            features[f"gen_user_{user_num}_q"] = float(q_value)

    return features
