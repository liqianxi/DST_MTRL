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


