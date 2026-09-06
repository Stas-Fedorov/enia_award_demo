"""Compact network loader for the ENIA demo.

Derived from the original simulator/Network_tested.py. Loads the rural
low-voltage grid, user load/generation profiles, and EV charging profiles
from a single daily Excel workbook.  AC power-flow results are used to
refine bus load set-points so that the demo reproduces the project workflow
without running the placement optimization.
"""

import math
import os
import re
from typing import Dict, List, Optional

import pandas as pd


class Network:
    """Lightweight network container used by the one-day demo."""

    # Fixed mapping used by the original project.  Station ids are mapped to
    # bus indices in the JSON configuration.
    CS_TO_BUS_MAP = {
        "cs_0": 2, "cs_1": 7, "cs_2": 3, "cs_3": 9, "cs_4": 10,
        "cs_5": 11, "cs_6": 13, "cs_7": 14, "cs_8": 15, "cs_9": 17,
        "cs_10": 21, "cs_11": 6, "cs_12": 20, "cs_13": 5,
    }

    def __init__(self, param_dict: Dict, base_dir: str, day_file: str):
        self.BUS_TO_CS_MAP = {v: k for k, v in self.CS_TO_BUS_MAP.items()}

        self.base_dir = base_dir
        self.param_dict = param_dict

        self.time_resolution = param_dict.get("time_resolution_min", 15)
        self.n_ts = 24 * 60 // self.time_resolution  # 96 for 15-min data
        self.tcf = param_dict.get("time_conversion_factor", 0.25)
        self.P_col = param_dict.get("P_col", 0.022)
        self.n_utenti = param_dict.get("n_utenti", 14)

        # Excel inputs.  The demo uses a single daily workbook that contains both
        # the user profiles ("Utente X") and the per-station EV profiles
        # ("Ricarica_cs_X").
        data_dir = os.path.join(base_dir, "data")
        self.input_excel_L_G = os.path.join(data_dir, day_file)
        self.input_ricarica = os.path.join(data_dir, day_file)

        # Network data structures are taken directly from the JSON configuration.
        self.bus_data = param_dict.get("bus_data", [])
        self.line_data = param_dict.get("line_data", [])
        self.load_data = param_dict.get("load_data", [])
        self.sgen_data = param_dict.get("sgen_data", [])
        self.transformer_data = param_dict.get("transformer_data", [])
        self.ext_grid_data = param_dict.get("ext_grid_data", [])

        self.v_min = param_dict.get("v_min", 0.9)
        self.v_max = param_dict.get("v_max", 1.1)
        self.load_max = param_dict.get("load_max", 100)

        # Filled by load_timeseries().
        self.charging_profiles: Dict[str, pd.Series] = {}
        self.available_stations: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_timeseries(self) -> None:
        """Load load/generation and EV profiles from the daily workbook."""
        if not os.path.isfile(self.input_excel_L_G):
            raise FileNotFoundError(f"Daily profile workbook not found: {self.input_excel_L_G}")

        # User profiles: one sheet per user, columns are unnamed in the original
        # files; the first column is time, the second is active power (Carico),
        # the third is generation (Gen).
        sheets = [f"Utente {i}" for i in range(1, self.n_utenti + 1)]
        data = self._import_user_profiles(self.input_excel_L_G, sheets)

        # EV profiles: sheets named "Ricarica_cs_X" with header in row 2.
        self._load_charging_profiles(self.input_ricarica)

        tanfi = math.tan(math.acos(0.99))

        utente_idx = 0
        for bus in self.load_data:
            if bus.get("empty", False):
                bus["p_mw"] = pd.Series([0.0] * self.n_ts)
                bus["q_mvar"] = pd.Series([0.0] * self.n_ts)
                continue

            if utente_idx < len(data):
                p_series = data[sheets[utente_idx]]["Carico"]
                q_series = [p * tanfi for p in p_series]
                bus["p_mw"] = pd.Series(p_series)
                bus["q_mvar"] = pd.Series(q_series)
                utente_idx += 1
            else:
                bus["p_mw"] = pd.Series([0.0] * self.n_ts)
                bus["q_mvar"] = pd.Series([0.0] * self.n_ts)

        utente_idx_g = 0
        for sgen in self.sgen_data:
            if sgen.get("empty", False):
                sgen["p_mw"] = pd.Series([0.0] * self.n_ts)
                sgen["q_mvar"] = pd.Series([0.0] * self.n_ts)
                continue

            if utente_idx_g < len(data):
                sgen["p_mw"] = pd.Series(data[sheets[utente_idx_g]]["Gen"])
                sgen["q_mvar"] = pd.Series([0.0] * self.n_ts)
                utente_idx_g += 1
            else:
                sgen["p_mw"] = pd.Series([0.0] * self.n_ts)
                sgen["q_mvar"] = pd.Series([0.0] * self.n_ts)

    def update_with_optimization_results(self) -> None:
        """Load the pre-computed AC power-flow result and update load set-points.

        The original workflow runs a placement optimization once and then uses
        the resulting AC_pf_22.0.xlsx file to refine load profiles.  The demo
        ships the same AC_pf file, so we simply replay that update.
        """
        ac_pf_file = os.path.join(self.base_dir, "data", "AC_pf_22.0.xlsx")
        if not os.path.isfile(ac_pf_file):
            raise FileNotFoundError(f"AC_pf file not found: {ac_pf_file}")

        loads_p_df = pd.read_excel(ac_pf_file, sheet_name="Loads_P", index_col=0)
        loads_q_df = pd.read_excel(ac_pf_file, sheet_name="Loads_Q", index_col=0)

        updated = 0
        for i, load in enumerate(self.load_data):
            bus_name = load.get("name", f"Bus {i} load")
            if load.get("empty", False):
                continue
            if bus_name in loads_p_df.columns and bus_name in loads_q_df.columns:
                self.load_data[i]["p_mw"] = pd.Series(loads_p_df[bus_name].values[: self.n_ts])
                self.load_data[i]["q_mvar"] = pd.Series(loads_q_df[bus_name].values[: self.n_ts])
                updated += 1

        if updated == 0:
            raise RuntimeError("No load profiles could be updated from AC_pf file")

    def station_to_bus(self, station_id: str) -> Optional[int]:
        return self.CS_TO_BUS_MAP.get(station_id)

    def bus_to_station(self, bus_idx: int) -> Optional[str]:
        return self.BUS_TO_CS_MAP.get(bus_idx)

    def get_charging_power_at_timestep(self, station_id: str, timestep: int) -> float:
        if station_id in self.charging_profiles:
            try:
                requested_kw = float(self.charging_profiles[station_id].iloc[timestep])
                return min(requested_kw, self.P_col * 1000.0)
            except (IndexError, KeyError):
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _import_user_profiles(self, file_name: str, sheet_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """Read the user sheets and return {sheet: {Carico: [...], Gen: [...]}}."""
        sheets = pd.read_excel(file_name, sheet_name=sheet_names, header=0)
        result = {}
        for sheet_name, df in sheets.items():
            # Drop fully-empty columns/rows and ensure we have at least three columns.
            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            df = df.iloc[: self.n_ts]
            cols = [c for c in df.columns if str(c).strip() != ""]
            if len(cols) < 3:
                raise ValueError(f"Sheet {sheet_name} has fewer than 3 data columns")
            result[sheet_name] = {
                "Carico": df.iloc[:, 1].astype(float).tolist(),
                "Gen": df.iloc[:, 2].astype(float).tolist(),
            }
        return result

    def _load_charging_profiles(self, file_name: str) -> None:
        self.charging_profiles = {}
        with pd.ExcelFile(file_name) as xls:
            sheet_names = xls.sheet_names

        ricarica_sheets = [s for s in sheet_names if s.startswith("Ricarica_cs_")]
        if not ricarica_sheets:
            raise ValueError(f"No Ricarica_cs_* sheets found in {file_name}")

        for sheet in ricarica_sheets:
            station_id = sheet.replace("Ricarica_", "")
            df = pd.read_excel(file_name, sheet_name=sheet, header=1)
            # The fourth column (index 3) is the power in kW.
            power_kw = df.iloc[:, 3].astype(float).reset_index(drop=True)
            self.charging_profiles[station_id] = power_kw

        self.available_stations = sorted(self.charging_profiles.keys())[:3]
        self.charging_profiles = {
            station_id: self.charging_profiles[station_id]
            for station_id in self.available_stations
        }
