#!/bin/bash


for ((i=0; i<=99; i++))
do
    python3 evaluate_trained.py --env_name=rware-small-15ag-v1 --path=results/trained_models/101/u196000 --seed=$i
    echo "Iteration $i finished"
done