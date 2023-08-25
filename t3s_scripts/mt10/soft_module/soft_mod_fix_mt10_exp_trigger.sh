#!/bin/bash

for seed in {0..9}; 
do
    sbatch soft_mod_fix_mt10_single.sh $seed
done