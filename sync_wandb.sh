#!/bin/bash

# Iterate through folders starting with "offline-"
for folder in /home/qianxi/scratch/sparse_training/sep_t3s/DST_RL/wandb/offline-*; do
    if [ -d "$folder" ]; then
        echo "Syncing $folder"
        wandb sync --include-offline "$folder"
    fi
done