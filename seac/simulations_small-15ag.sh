#!/bin/bash


for ((i=0; i<=99; i++))
do
    python3 evaluate_cbs_rware_replanning.py --env_name=rware-small-15ag-v1 --seed=$i
    echo "Iteration $i finished"
done