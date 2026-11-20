# mmWave Beam Selection — Generalization & Robustness Study

Code for a course project on sub-6 GHz-assisted mmWave beam prediction using DeepMIMO.

## Research contributions

1. **Angle 1 — Generalization boundary:** How does prediction accuracy degrade when the test environment differs from the training environment?
2. **Angle 5 — Noise robustness:** How gracefully does the model degrade when sub-6 GHz input is noisy?

## Project structure

```
mmwave_project/
├── data/            # DeepMIMO loaders and preprocessing
├── models/          # Neural network architectures
├── experiments/     # Experiment scripts (Angle 1, Angle 5)
├── utils/           # Metrics, environmental distance, plotting
├── scripts/         # Entry points to run experiments
└── results/         # Saved checkpoints, plots, CSVs
```

## Setup

```bash
# Virtual env (recommended on HPC)
python3 -m venv ~/mmwave-env
source ~/mmwave-env/bin/activate

pip install deepmimo torch numpy scipy matplotlib pandas tqdm seaborn
```

**DeepMIMO v4 auto-downloads scenarios** — no manual ZIP downloads needed.
The scenes used are:
- `asu_campus_3p5` + `asu_campus_28` (ASU scene, sub-6 and mmWave)
- `city_6_miami_3p5` + `city_6_miami_28` (Miami scene, sub-6 and mmWave)

## Run

```bash
# Step 0 (do this first!): verify DeepMIMO works on your machine
python scripts/00_check_deepmimo.py

# Step 1: Generate datasets (downloads + preprocesses + caches)
python scripts/01_generate_data.py

# Step 2: Train baseline model (reproduces Alrabeiah 2020)
python scripts/02_train_baseline.py

# Step 3: Angle 1 — cross-scenario generalization experiments
python scripts/03_experiment_generalization.py

# Step 4: Angle 5 — noise robustness experiments
python scripts/04_experiment_robustness.py

# Step 5: Generate all plots and summary tables
python scripts/05_make_plots.py
```
