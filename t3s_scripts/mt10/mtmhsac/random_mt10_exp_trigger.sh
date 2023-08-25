#!/bin/bash

for seed in {0..9}; 
do
    sbatch random_mt10_single.sh $seed
done