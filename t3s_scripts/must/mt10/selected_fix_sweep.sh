#!/bin/bash

# first, don't use SL loss.
for seed in {3..4}; 
do

    for pruning_ratio in {0.1,0.7};
    do

        # First, more than 1 task.
        for selected_task in {2,4,10};
        do 
            for itv in {50,100,200,400}
            do
                sbatch selected_fix_sweep_single.sh $seed $itv $pruning_ratio 1e-3 0 0 $selected_task 1
            done
        done

        # Second, 1 task.
        for task_id in {0..9}
        do
            for itv in {50,100,200,400}
            do
                sbatch selected_fix_sweep_single.sh $seed $itv $pruning_ratio 1e-3 0 0 1 $task_id
            done
        done
    done


done

# Second, use SL loss and RL loss
for seed in {3..4}; 
do

    for pruning_ratio in {0.1,0.7};
    do
        for gen_lr in {1e-3,1e-2};
        do
            for use_trajectory_info in {0,1};
            do
                for selected_task in {2,4,10};
                do 
                    for itv in {50,100,200,400}
                    do
                        sbatch selected_fix_sweep_single.sh $seed $itv $pruning_ratio $gen_lr 1 $use_trajectory_info $selected_task 1
                    done
                done
            done

        done
    done

done
