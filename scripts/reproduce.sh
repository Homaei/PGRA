#!/usr/bin/env bash
# End-to-end reproduction of the PGRA experiments.
#
# Prerequisites:
#   * Python >= 3.10 with the packages listed in requirements.txt
#   * Raw BATADAL and WADI CSVs placed at:
#       pgra/data/raw/BATADAL_dataset03.csv
#       pgra/data/raw/BATADAL_dataset04.csv
#       pgra/data/raw/BATADAL_network.inp
#       pgra/data/raw/WADI_14days_new.csv
#       pgra/data/raw/WADI_attackdataT.csv
#
# Usage:
#   bash scripts/reproduce.sh

set -euo pipefail

# Pre-processing
python -m pgra.data.batadal_processor
python -m pgra.data.wadi_processor
python -m pgra.data.partition

# Main comparison on both datasets
python -m pgra.experiments.run_main --dataset batadal
python -m pgra.experiments.run_main --dataset wadi

# Ablations and supporting analyses
python -m pgra.experiments.run_ablation     --dataset batadal
python -m pgra.experiments.run_stealthiness --dataset batadal --seed 42
python -m pgra.experiments.run_convexity    --dataset batadal --seed 42
python -m pgra.experiments.run_edge_feasibility --dataset batadal

# Consolidate everything into a single report
python -m pgra.experiments.finalize
