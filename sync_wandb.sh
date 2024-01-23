#!/bin/bash


for folder in /home/qianxi/scratch/sparse_training/dec_must/DST_RL/wandb/offline-*; do
    if [ -d "$folder" ]; then
        echo "Syncing $folder"
        wandb sync "$folder"
        sleep 3
    fi
done
