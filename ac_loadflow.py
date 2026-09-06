"""Compact AC load-flow wrapper.

Reuses the implementation in f_AC_ldf_verifica.py and returns the same
result dictionary used by the rest of the demo.
"""

import warnings

import numpy as np
import pandas as pd
import pypsa


def run_ac_load_flow(bus_data, load_data, sgen_data, line_data, ext_grid_data,
                     n_ts, tcf, transformers_data):
    """Run an AC power flow for a single or multi-timestep snapshot.

    Parameters are passed directly to a PyPSA Network built from the project
    configuration.  The function is intentionally kept close to the original
    f_AC_ldf_verifica.py implementation.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", module="pypsa.components")

    grid = pypsa.Network(name="Network BT")
    grid.set_snapshots(range(n_ts))
    grid.add("Carrier", "AC")

    for bus in bus_data:
        grid.add("Bus", name=bus["name"], v_nom=bus["v_nom"], carrier="AC")

    bus_names = {i: bus["name"] for i, bus in enumerate(bus_data)}

    for line in line_data:
        if line == line_data[-1]:
            s_nom = 11 * line["max_i_ka"] * (3 ** 0.5)
        else:
            s_nom = 0.4 * line["max_i_ka"] * (3 ** 0.5)
        grid.add(
            "Line",
            name=line["name"],
            bus0=bus_names[line["from_bus"]],
            bus1=bus_names[line["to_bus"]],
            carrier="AC",
            x=line["x_ohm_per_km"] * line["length_km"],
            r=line["r_ohm_per_km"] * line["length_km"],
            length=line["length_km"],
            s_nom=s_nom,
        )

    for sgen in sgen_data:
        grid.add(
            "Generator",
            name=sgen["name"],
            bus=bus_names[sgen["bus_index"]],
            carrier="AC",
            control=sgen["control"],
            p_set=sgen["p_mw"],
            q_set=sgen["q_mvar"],
        )

    for load in load_data:
        grid.add(
            "Load",
            name=load["name"],
            bus=bus_names[load["bus_index"]],
            carrier="AC",
            p_set=load["p_mw"],
            q_set=load["q_mvar"],
        )

    for ext_grid in ext_grid_data:
        grid.add(
            "Generator",
            name=ext_grid["name"],
            bus=bus_names[ext_grid["bus_index"]],
            carrier="AC",
            control="Slack",
            p_set=ext_grid["max_p_mw"],
            q_set=ext_grid["max_q_mvar"],
        )

    for tr in transformers_data:
        grid.add(
            "Transformer",
            name=tr["name"],
            bus0=bus_names[tr["bus0"]],
            bus1=bus_names[tr["bus1"]],
            carrier="AC",
            model=tr["model"],
            v_nom_0=tr["v_nom_0"],
            v_nom_1=tr["v_nom_1"],
            f_nom=tr["f_nom"],
            vsc=tr["vsc"],
            vscr=tr["vscr"],
            pfe=tr["pfe"],
            i0=tr["i0"],
            phase_shift=tr["phase_shift"],
            x=tr["x"],
            r=tr["r"],
            s_nom=tr["s_nom"],
            tap_side=tr["tap_side"],
            tap_neutral=tr["tap_neutral"],
            tap_min=tr["tap_min"],
            tap_max=tr["tap_max"],
            tap_step=tr["tap_step"],
        )

    grid.pf()

    a = grid.lines_t.p0
    b = grid.lines_t.p1
    p_losses = (a + b).clip(lower=0)
    S_line = (grid.lines_t.p0 ** 2 + grid.lines_t.q0 ** 2) ** 0.5
    loading_percentage = (S_line / grid.lines.s_nom) * 100

    Cabina_power_ts = grid.buses_t.p.loc[:, "Bus Monte MT"]
    Cabina_power = pd.DataFrame(Cabina_power_ts)
    Cabina_energy = Cabina_power * tcf
    Cabina_power_fil = Cabina_energy[Cabina_energy >= 0]
    Cabina_power_neg = Cabina_energy[Cabina_energy <= 0]
    Cabina_energy_pos = Cabina_power_fil.sum().sum()
    Cabina_energy_neg = Cabina_power_neg.sum().sum()

    return {
        "Bus_V_Mag_PU": grid.buses_t.v_mag_pu,
        "Bus_V_Ang": grid.buses_t.v_ang,
        "Bus_P": grid.buses_t.p,
        "Bus_Q": grid.buses_t.q,
        "Lines_P0": grid.lines_t.p0,
        "Lines_Q0": grid.lines_t.q0,
        "Lines_P1": grid.lines_t.p1,
        "Lines_Q1": grid.lines_t.q1,
        "Generators_P": grid.generators_t.p,
        "Generators_Q": grid.generators_t.q,
        "Loads_P": grid.loads_t.p,
        "Loads_Q": grid.loads_t.q,
        "p_losses": p_losses,
        "q_losses": (grid.lines_t.q0 + grid.lines_t.q1).clip(lower=0),
        "loading percentage": loading_percentage,
        "Cabina_energy": Cabina_energy_pos,
        "Cabina_power": Cabina_power,
        "Cabina_Energy_neg": Cabina_energy_neg,
    }
