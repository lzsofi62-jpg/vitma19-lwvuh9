#!/bin/bash
set -e
echo "Running data processing..."
python 01_data_processing.py
echo "Running baseline model..."
python 02_baseline.py
echo "Running model training..."
python 03_train.py
echo "Running evaluation..."
python 04_evaluation.py
echo "Pipeline finished successfully."