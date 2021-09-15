#!/bin/bash


for ((i=0; i<=99; i++))
do
    python3 evaluate_trained.py --env_name=rware-large-15ag-v1 --path=results/trained_models/103/u220000 --seed=$i
    echo "Iteration $i finished"
done