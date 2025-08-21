#!/bin/bash

for n in $(seq 0 5 45); do
  echo "Running round_$n..."
  python Applications/Empir3D_Framework/gpd_eval_app.py --config Config/round_${n}.json
done