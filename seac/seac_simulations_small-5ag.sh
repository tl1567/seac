#!/bin/bash


for ((i=0; i<=99; i++))
do
    python3 evaluate_trained.py --env_name=rware-small-5ag-v1 --path=results/trained_models/95/u316000 --seed=$i
    echo "Iteration $i finished"
done