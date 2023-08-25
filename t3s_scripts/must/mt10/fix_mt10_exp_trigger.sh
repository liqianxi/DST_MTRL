#!/bin/bash

for seed in {0..2}; 
do
    sbatch fix_mt10_single.sh $seed
done