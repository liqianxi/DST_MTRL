import sys
# import sys
sys.path.append(".") 

import torch

import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.hyperpolicies as policies
import torchrl.hypernetworks as networks
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSAC
from torchrl.collector.para import ParallelCollector
from torchrl.collector.para import AsyncParallelCollector
from torchrl.collector.para.mt import SingleTaskParallelCollectorBase
from torchrl.collector.para.async_mt import AsyncSingleTaskParallelCollector
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorUniform

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym

from metaworld_utils.meta_env import get_meta_env
#import wandb
import pickle


RESTORE = int(os.getenv('RESTORE', '0'))

CPU_NUM = 1
os.environ['OMP_NUM_THREADS'] = str(CPU_NUM)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_NUM)
os.environ['MKL_NUM_THREADS'] = str(CPU_NUM)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(CPU_NUM)
os.environ['NUMEXPR_NUM_THREADS'] = str(CPU_NUM)
torch.set_num_threads(CPU_NUM)


def experiment(args):
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    print(device)
    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from torchrl.networks.init import normal_init

    pf = policies.GuassianContPolicy(
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        **params['net'] )
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )
    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )

    example_ob = env.reset()
    # print(example_ob, example_ob.shape)

    example_dict = {
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0]
    }
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    if RESTORE:
        with open(osp.join(osp.join(logger.work_dir,"model"), "replay_buffer.pkl"), 'rb') as f:
            replay_buffer = pickle.load(f)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    print(env.action_space)
    print(env.observation_space)
    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = MTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
