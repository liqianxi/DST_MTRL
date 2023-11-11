
import sys
sys.path.append(".")

import torch
import wandb
import os,json
#os.system("module load python/3.8")

import time,copy
import os.path as osp

import numpy as np
import torch.multiprocessing as mp
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSAC, MUST_SAC
from torchrl.collector.para import ParallelCollector
from torchrl.collector.para import AsyncParallelCollector
from torchrl.collector.para.mt import SingleTaskParallelCollectorBase
from torchrl.collector.para.async_mt import AsyncSingleTaskParallelCollector
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorUniform

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym

from metaworld_utils.meta_env import get_meta_env

import random
import pickle

torch.set_printoptions(profile="full")

#torch.autograd.set_detect_anomaly(True)
RESTORE = int(os.getenv('RESTORE', '0'))

CPU_NUM = 1
os.environ['OMP_NUM_THREADS'] = str(CPU_NUM)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_NUM)
os.environ['MKL_NUM_THREADS'] = str(CPU_NUM)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(CPU_NUM)
os.environ['NUMEXPR_NUM_THREADS'] = str(CPU_NUM)
torch.set_num_threads(CPU_NUM)

def random_initialize_masks(network, pruning_ratio):
    neuron_mask_list = []
<<<<<<< HEAD
    all = [i for i in network.base.fcs] + [network.last]

    #print("network",network)
    for each_layer in all:
        weight_shape = each_layer.weight.shape
        neurons = weight_shape[0]*weight_shape[1]
=======

    #print("network",network)
    for each_layer in network.base.fcs:
        neurons = each_layer.bias.shape[0]
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
        #print("neurons",neurons)
        #print("ratio",pruning_ratio)
        neuron_mask = torch.zeros(neurons)

<<<<<<< HEAD
        #print("ones",ones)
        idx = torch.randperm(neurons)[:int(neurons - neurons * pruning_ratio)]
        neuron_mask[idx] = 1
        neuron_mask_list.append(neuron_mask.reshape(weight_shape))

        bias_shape = each_layer.bias.shape

        neurons = bias_shape[0]
        #print("neurons",neurons)
        #print("ratio",pruning_ratio)
        neuron_mask = torch.zeros(neurons)

        #print("ones",ones)
        idx = torch.randperm(neurons)[:int(neurons - neurons * pruning_ratio)]
=======
        tmp = neurons - neurons * pruning_ratio
        #print("tmp",tmp)
        ones = int(tmp)
        #print("ones",ones)
        idx = torch.randperm(neurons)[:ones]
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
        neuron_mask[idx] = 1
        neuron_mask_list.append(neuron_mask)

    return neuron_mask_list




def experiment(args):


    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])
    pruning_ratio = args.pruning_ratio
<<<<<<< HEAD
    mask_update_interval = args.mask_update_interval
    params['sparse_training']["pruning_ratio"] = args.pruning_ratio
    #params['general_setting']["update_end_epoch"] = args.mask_end_update_episode
    params['general_setting']["mask_update_interval"] = args.mask_update_interval
    params["general_setting"]["sl_optim_times"] = args.sl_optim_times
    params["general_setting"]["generator_lr"] = args.generator_lr
    params["general_setting"]["use_trajectory_info"] = args.use_trajectory_info
    params["general_setting"]["use_sl_loss"] = args.use_sl_loss
    

    #print("args.success_traj_update_only",args.success_traj_update_only)
    group_name = args.wandb_group_name
=======
    params['sparse_training']["pruning_ratio"] = args.pruning_ratio
    # params['general_setting']["mask_update_interval"] = args.mask_update_interval
    # params['general_setting']["update_end_epoch"] = args.mask_end_update_episode
    group_name = "0817_sweep"
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e

    #print("pruning_ratio",pruning_ratio)


    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
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


    
    mp.set_start_method('spawn', force=True)

    example_ob = env.reset()

    example_embedding = env.active_task_one_hot

    embedding_shape = np.prod(example_embedding.shape)
    # Initialize policy net.
    # The policy network 
    # Input: S,onehot(task)

    # (19,) -> obs + goal obs + one-hot
    # (4,)

    pf = policies.EmbedGuassianContPolicy(
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        **params['net'])

    print("finish policy net init")

    # if args.pf_snap is not None:
    #     pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    # Initialize Q1 and Q2 net, the initialization of Q1_target and Q2_target 
    # will be in TwinSACQ.
    # Input: S,A,onehot(task)
    qf1 = networks.MaskedNet( 
        input_shape = env.observation_space.shape[0] 
                    + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )
    qf2 = networks.MaskedNet( 
        input_shape = env.observation_space.shape[0]
                    + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )

    # Initialize VAE model.
    encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
<<<<<<< HEAD
                                         params['traj_encoder']["latent_size"],
                                          device=device).to(device)

    q1_encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
                                         params['traj_encoder']["latent_size"],
                                          device=device).to(device)

    q2_encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
                                         params['traj_encoder']["latent_size"],
                                          device=device).to(device)
=======
                                         params['traj_encoder']["latent_size"], device=device).to(device)

    q1_encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
                                         params['traj_encoder']["latent_size"], device=device).to(device)

    q2_encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
                                         params['traj_encoder']["latent_size"], device=device).to(device)
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e

    # Initialize Mask generators.
    # For Policy net, Q1, Q2, we need 3 mask generators.
    # We don't need separate mask generator for target networks, since the
    # target nets should follow the gradients of Q nets, but with discount factor.
    # We use the same mask with their corresponding Q networks.

    policy_mask_generator = networks.MaskGeneratorNet(
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        pruning_ratio=pruning_ratio,
        device=device,
        info_dim=params['traj_encoder']["latent_size"],
<<<<<<< HEAD
        trajectory_encoder=encoder,
        main_input_dim=env.observation_space.shape[0],
        main_out_dim=2 * env.action_space.shape[0],
        use_trajectory_info=args.use_trajectory_info)
=======
        trajectory_encoder=encoder
        )

    # test_traj = torch.randn((4,20,19)).to(device)
    # onehot = torch.zeros((4,1,10)).to(device)
    # onehot[0][0][0] = 1
    # onehot[1][0][1] = 1
    # onehot[2][0][2] = 1
    # onehot[3][0][3] = 1




    # policy_mask_generator(test_traj,onehot)
    #assert 1==2
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e

    qf1_mask_generator = networks.MaskGeneratorNet(
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        pruning_ratio=pruning_ratio,
        device=device,
        info_dim=params['traj_encoder']["latent_size"],
<<<<<<< HEAD
        trajectory_encoder=q1_encoder,
        main_input_dim=env.observation_space.shape[0] 
                      + env.action_space.shape[0],
        main_out_dim=1,
        use_trajectory_info=args.use_trajectory_info
=======
        trajectory_encoder=q1_encoder
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
        )
    qf2_mask_generator = networks.MaskGeneratorNet(
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        pruning_ratio=pruning_ratio,
        device=device,
        info_dim=params['traj_encoder']["latent_size"],
<<<<<<< HEAD
        trajectory_encoder=q2_encoder,
        main_input_dim=env.observation_space.shape[0] 
                      + env.action_space.shape[0],
        main_out_dim=1,
        use_trajectory_info=args.use_trajectory_info)
=======
        trajectory_encoder=q2_encoder)
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
    
    print("mask generator finish initialization")

    

    # if args.qf1_snap is not None:
    #     qf1.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    # if args.qf2_snap is not None:
    #     qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "embedding_inputs": example_embedding
    }

    replay_buffer = AsyncSharedReplayBuffer(int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    manager = mp.Manager()
    # This is used to train the encoder and compare traj similarity.
    state_trajectory = manager.dict()

    for task_idx in range( env.num_tasks):
        state_trajectory[task_idx] = []

<<<<<<< HEAD

    traj_collect_mod = manager.dict()

    for task_idx in range( env.num_tasks):
        traj_collect_mod[task_idx] = 0

=======
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
    # Mask buffer, stores the current masks for each layer, 
    # for each task and for each network type(Q1, Q2, policy).
    # Initialize the binary mask for all the weights and bias, make sure
    # follow the pruning ratio requirement.

    all_mask_buffer = {}

    for net_type in ["Q1","Q2","Policy"]:
        mask_buffer = manager.dict()

        for task_idx in range( env.num_tasks):
            mask_buffer[task_idx] = []

            if net_type == "Q1":
                net = qf1
            elif net_type == "Q2":
                net = qf2 
            elif net_type == "Policy":
                net = pf

            neuron_masks = random_initialize_masks(net, pruning_ratio)
<<<<<<< HEAD

            mask_buffer[task_idx] = neuron_masks

        all_mask_buffer[net_type] = mask_buffer   

    final_mask = {"Q1":{},"Q2":{},"Policy":{}} 



=======
            #print("neuron_masks",neuron_masks)
            mask_buffer[task_idx] = neuron_masks

        all_mask_buffer[net_type] = mask_buffer       

    #print(all_mask_buffer["Policy"])

    #assert 1==2
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
    if RESTORE:
        with open(osp.join(osp.join(logger.work_dir,"model"), "replay_buffer.pkl"), 'rb') as f:
            replay_buffer = pickle.load(f)

    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['state_trajectory'] = state_trajectory
    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,state_trajectory=state_trajectory,
<<<<<<< HEAD
        traj_collect_mod=traj_collect_mod,
=======
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
        mask_buffer=all_mask_buffer["Policy"],
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        manager=manager,
        pretraining_epoch = params['general_setting']["pretrain_epochs"],
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    
    agent = MUST_SAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        mask_generators = {"policy_mask_generator": policy_mask_generator,
                           "qf1_mask_generator": qf1_mask_generator,
                           "qf2_mask_generator": qf2_mask_generator},
        task_nums=env.num_tasks,
        trajectory_encoder=encoder,
<<<<<<< HEAD
        traj_collect_mod=traj_collect_mod,
        mask_buffer=all_mask_buffer,
        final_mask=final_mask,
        mask_update_itv=mask_update_interval,
=======
        mask_buffer=all_mask_buffer,
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
        **params['sac'],
        **params['general_setting']
    )

    agent.train(env.num_tasks,params,group_name)


if __name__ == "__main__":
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    experiment(args)
    
    #print(prof)