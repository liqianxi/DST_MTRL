#!/bin/bash

for seed in {0..2}; 
do
    for mask_update_interval in {10,25,50,100};
    do
        for pruning_ratio in {0.5,0.3,0.1,0.7};
        do

            sbatch fix_sweep_single.sh $seed $mask_update_interval $pruning_ratio

        done

    done
done
