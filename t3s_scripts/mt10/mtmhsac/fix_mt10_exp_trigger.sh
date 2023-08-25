#!/bin/bash

for seed in {0..9}; 
do
    sbatch fix_mt10_single.sh $seed
done