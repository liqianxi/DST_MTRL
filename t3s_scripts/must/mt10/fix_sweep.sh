#!/bin/bash

for seed in {0..2}; 
do

    for pruning_ratio in {0.5,0.1,0.7};
    do
        for gen_lr in {1e-3,5e-3,1e-2};
        do

            sbatch fix_sweep_single.sh $seed 25 $pruning_ratio $gen_lr 0 0

        done
    done


done
for seed in {0..2}; 
do

    for pruning_ratio in {0.5,0.1,0.7};
    do
        for gen_lr in {1e-3,5e-3,1e-2};
        do

            for use_trajectory_info in {0,1};
            do
                sbatch fix_sweep_single.sh $seed 25 $pruning_ratio $gen_lr 1 $use_trajectory_info
            done

        done
    done

done
