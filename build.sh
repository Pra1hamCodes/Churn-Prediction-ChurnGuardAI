#!/bin/bash
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Training model ==="
python churn_prediction.py

echo "=== Build complete ==="
