#!/bin/bash

# first, don't use SL loss.
for seed in {3..4}; 
do

    for pruning_ratio in {0.3,0.7};
    do

        # Second, 1 task.
        for task_id in {0..2}
        do
            for itv in {50,100,400}
            do
                sbatch selected_fix_sweep_single.sh $seed $itv $pruning_ratio 1e-3 0 0 1 $task_id
            done
        done
    done

done