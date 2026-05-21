#!/bin/bash

# chmod +x experiments.sh
# ./experiments.sh

echo "===== Training ====="
python train.py

echo "===== Evaluating ====="
python evaluate.py

echo "===== Done ====="
echo "Results saved in results/"