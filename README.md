# ENIA One-Day EV Charging Control Demo

This compact example demonstrates one day of counterfactual EV charging control on the 22-bus Fforchneol Farm low-voltage network.

The demo loads:

- 96 snapshots at 15-minute resolution from an eventful test day;
- three EV charging stations, each capped at its 22 kW nominal rating;
- a pretrained Random Forest stability model;
- the network topology and fixed planning-stage set-points.

It first computes the uncontrolled day. When AC power flow detects a voltage or line-loading violation, the controller evaluates interpretable counterfactual charging assignments. It selects the highest-power assignment that is both predicted stable by the Random Forest and verified by AC power flow.

SUMO generation and model training are intentionally excluded so the example runs quickly and deterministically.

## Install

- Python 3.9-3.11
- Dependencies in `requirements.txt`

Create a virtual environment and install the packages:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

No Gurobi installation or license is required for this compact demonstrator. It performs a small discrete counterfactual search over candidate controls. The full research implementation uses a Gurobi MILP to embed the trained model and optimize continuous station-level controls.

## Launch test

The demo run is the end-to-end test:

```bash
python main.py
```

To run a different daily profile, pass `--day`:

```bash
python main.py --day Day_004_profiles.xlsx
```

## Check that it is working

If the run succeeds you will see log lines similar to:

```text
Baseline stable timesteps: 94/96
Controlled stable timesteps: 96/96
Intervention timesteps: [30, 73]
Average charging power: baseline 28.45 kW, controlled 28.39 kW
```

(Exact numbers may shift slightly depending on the selected day file.)

Each run overwrites:

- `results/one_day_control.png`: baseline versus controlled stability, voltage, loading, and charging power;
- `results/one_day_metrics.csv`: per-timestep results;
- `results/one_day_summary.json`: daily summary;
- `demo.log`: concise execution log.

The reference run identifies two unstable baseline intervals and restores stability in both with minimal impact on average delivered charging power.

## Repository structure

```text
main.py                 one-command demo and result plotting
controller.py           counterfactual candidate evaluation and control
features.py             pretrained-model feature construction
network.py              compact network and daily-profile loader
ac_loadflow.py          PyPSA AC power-flow wrapper
data/                    one-day profile, network configuration, set-points
models/                  pretrained Random Forest and metadata
results/                 reference output figure and metrics
```

## Sharing and licensing

No software or data license is asserted by this demo. Before publishing the repository, confirm that the project team has the right to redistribute the network data, daily profile, pretrained model, and derived set-points.

## Scope

This is a transparent, minimal reproduction of the operational stability module prepared for evaluation. It is not a production grid controller. The complete research code uses broader training data, continuous MILP optimization, sensitivity models, benchmarking, and multi-day evaluation.
