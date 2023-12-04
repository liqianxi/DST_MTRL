# MUST


The way to install metaworld and other dependencies can be found at here:https://github.com/yuyuanq/T3S-MTRL-Pytorch  

## Repo Structure

/log : contains all the logged data and training model weights.
/starter : this folder contains the entry function of several different multi-task RL methods, since I write my code based on other people's code, some of these methods may no longer be usable, our framework's file is mt_must_sac.py.

/meta_config/must_configs : the configs I'm using for my framework.

/torchrl : the folder that contains all the components of our framework.
/torchrl/rl_algo.py and /torchrl/off_policy/must_sac.py : the first one contains the full training loop, mask update. the second one contains how our modified version of SAC do the update. Some other files in the same folder may contains parent classes of our algorithm.

/torchrl/collector/async_mt.py : important, this file contains the logic how we spawn a couple of processes to actually interact with each multitask environment in parallel, they collect trajectory and save tuples to the shared buffer. Some other files in the same folder may contains parent classes of our collector.

/torchrl/env : some wrappers and utils, I never touch these.

/torchrl/networks/nets.py : important, defines my mask generator and base network.

/torchrl/networks/trajectory_encoder.py : important, defines the trajectory encoder (lstm), we are not using VAE loss at this moment so some methods are not in used.

/torchrl/policies : policy network related.

/torchrl/replay_buffers : replay buffer, as indicated by the name.

/torchrl/otherfiles : usually I don't modify other files.

/t3s_scripts/must : the bash scripts I use to submit jobs on compute canada.

## Install

1. Use python3.8;  
2. run `git clone --single-branch --branch accurate_mask_revise https://github.com/liqianxi/DST_RL.git` to clone my work branch.  
3. Find req.txt in the root. In your virtual env, use `pip install -r req.txt`.
4. Find a location outside DST_RL, install metaworld:
`git clone https://github.com/RchalYang/metaworld.git ;
cd metaworld ;pip install -e .;

5. Install mujoco: go to https://www.roboti.us/download.html and download mujoco200, add this to your .bashrc: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qianxi/.mujoco/mujoco200/bin`, replace my own path with your mujoco200/bin file path. DOn't forget to run `source .bashrc` after you modify your bashrc.

## Run

1.Go to `/DST_RL/t3s_scripts/must/mt10/`. 
2. Open run_review.sh, change `cd /home/qianxi/scratch/sparse_training/dec_must/DST_RL` to your DST_RL root address.  
3. Run format:  
`bash run_review.sh $seed $update_interval $pruning_ratio $mask_generator_learning_rate $Use_supervised_loss $use_trajectory_info $selected_task`

4. For example:  
`bash run_review.sh 3 50 0.5 1e-4 1 1 10`
will run with random seed 3, the mask update interval is 50 iterations, pruning ratio=0.5, mask_generator_learning_rate is 1e-4 for the supervised loss, $Use_supervised_loss=1 means we use supervised loss and RL loss, $use_trajectory_info=1 means the mask generator also requires task trajectory as the input to generate the mask, $selected_task=10 means all 10 tasks in Metaworld MT10 benchmark will be used.


