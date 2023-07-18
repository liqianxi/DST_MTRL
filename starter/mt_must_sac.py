import sys
sys.path.append(".")

import torch

import os,json
import time,copy
import os.path as osp

import numpy as np

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



RESTORE = int(os.getenv('RESTORE', '0'))

CPU_NUM = 1
os.environ['OMP_NUM_THREADS'] = str(CPU_NUM)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_NUM)
os.environ['MKL_NUM_THREADS'] = str(CPU_NUM)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(CPU_NUM)
os.environ['NUMEXPR_NUM_THREADS'] = str(CPU_NUM)
torch.set_num_threads(CPU_NUM)

# def convert_neuron_masks_to_weight_mask(weight, neuron_masks, mode):
#     assert mode in ["row","col"], "Must specify an operation"
#     full_mask = weight

#     if mode == "row":
#         full_mask = torch.mul(full_mask.T, neuron_masks).T

#     else: 
#         full_mask = torch.mul(full_mask, neuron_masks)

#     return full_mask

def random_initialize_masks(network, pruning_ratio):
    neuron_mask_list = []
    all_layer_weight_shape = []
    # print(network.base.fcs) [Linear(in_features=33, out_features=400, bias=True), 
    # Linear(in_features=400, out_features=400, bias=True)]
    for each_layer in network.base.fcs:
        neurons = each_layer.bias.shape[0]
        all_layer_weight_shape.append(each_layer.weight.shape)
        neuron_mask = torch.zeros(neurons)
        ones = int(neurons * (1 - pruning_ratio))
        idx = torch.randperm(neurons)[:ones]
        neuron_mask[idx] = 1
        neuron_mask_list.append(neuron_mask)

    all_layer_weight_shape.append(network.last.weight.shape)

    #weight_
    # weight_array = [torch.ones(i) for i in all_layer_weight_shape]


    # for idx in range(len(neuron_mask_list)):
    #     pre_layer_weight = convert_neuron_masks_to_weight_mask(weight_array[idx], 
    #                                                            neuron_mask_list[idx], 
    #                                                            mode="row")
    #     weight_array[idx] = pre_layer_weight

    #     post_layer_weight = convert_neuron_masks_to_weight_mask(weight_array[idx+1], 
    #                                                             neuron_mask_list[idx], 
    #                                                             mode="col")
    #     weight_array[idx+1] = post_layer_weight

    #bias_mask_array = neuron_mask_list

    # So, for this task, mask out some neurons from the weight and bias tensors.
    # The mask of tensor is equivalent to the mask of neurons.
    # The mask of weight is applying neuron mask along rows or columns of the weight matrix, depends
    # on which matrix this neuron is affecting.
    return neuron_mask_list



        


def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")


    """
    {'reach-v1': <class 'metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place.SawyerReachPushPickPlaceEnv'>, 
    'push-v1': <class 'metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place.SawyerReachPushPickPlaceEnv'>, }'
    
    """
    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])
    pruning_ratio = params["sparse_training"]["pruning_ratio"]

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


    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from torchrl.networks.init import normal_init

    example_ob = env.reset()
    example_embedding = env.active_task_one_hot

    embedding_shape = np.prod(example_embedding.shape)
    # Initialize policy net.
    # The policy network 
    # Input: S,onehot(task)

    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    # (19,)
    # (4,)

    # TODO: allow random goal.
    pf = policies.EmbedGuassianContPolicy(
        input_shape = env.observation_space.shape[0] + embedding_shape, 
        output_shape = 2 * env.action_space.shape[0],
        **params['net'])

    print("finish policy net init")

    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    # Initialize Q1 and Q2 net, the initialization of Q1_target and Q2_target 
    # will be in TwinSACQ.
    # Input: S,A,onehot(task)
    qf1 = networks.MaskedNet( 
        input_shape = env.observation_space.shape[0] 
                    + env.action_space.shape[0] 
                    + embedding_shape,
        output_shape = 1,
        **params['net'] )
    qf2 = networks.MaskedNet( 
        input_shape = env.observation_space.shape[0]
                    + env.action_space.shape[0] 
                    + embedding_shape,
        output_shape = 1,
        **params['net'] )
    

    # Initialize VAE model.
    encoder = networks.TrajectoryEncoder(env.observation_space.shape[0],
                                         params['traj_encoder']["latent_size"], device=device).to("cpu")

    # Initialize Mask generators.
    # For Policy net, Q1, Q2, we need 3 mask generators.
    # We don't need separate mask generator for target networks, since the
    # target nets should follow the gradients of Q nets, but with discount factor.
    # We use the same mask with their corresponding Q networks.

    #print(params['net']["hidden_shapes"])
    policy_mask_generator = networks.MaskGeneratorNet(
        base_type=networks.MLPBase,
        traj_encoder_hidden_shape = params['traj_encoder']['hidden_shapes'],
        input_shape=env.observation_space.shape[0] * params['general_setting']["epoch_frames"],
        em_hidden_shapes=params['task_embedding']['em_hidden_shapes'],
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        trajectory_encoder=encoder,
        pruning_ratio=pruning_ratio,
        device="cpu"
        )

    # obs = torch.from_numpy(env.observation_space.sample()).float()
    # obs = torch.cat([obs]*params['general_setting']["epoch_frames"]).float()
    # emb = torch.from_numpy(example_embedding).float()

    #print([module for module in policy_mask_generator.modules()])
    #print(obs.shape, emb.shape) # (19,) (10,)
    #mask_list = policy_mask_generator(obs,emb)
    #print("mask_list",mask_list)
    """
    [MaskGeneratorNet(
    (base): MLPBase(
        (fc0): Linear(in_features=3800, out_features=400, bias=True)
        (fc1): Linear(in_features=400, out_features=400, bias=True)
    )
    (em_base): MLPBase(
        (fc0): Linear(in_features=10, out_features=400, bias=True)
        (fc1): Linear(in_features=400, out_features=400, bias=True)
    )
    (gating_weight_fc_0): Linear(in_features=400, out_features=400, bias=True)
    (gating_weight_cond_last): Linear(in_features=400, out_features=400, bias=True)
    (gating_weight_last): Linear(in_features=400, out_features=400, bias=True)
    ), MLPBase(
    (fc0): Linear(in_features=3800, out_features=400, bias=True)
    (fc1): Linear(in_features=400, out_features=400, bias=True)
    ), Linear(in_features=3800, out_features=400, bias=True), Linear(in_features=400, out_features=400, bias=True), MLPBase(
    (fc0): Linear(in_features=10, out_features=400, bias=True)
    (fc1): Linear(in_features=400, out_features=400, bias=True)
    ), Linear(in_features=10, out_features=400, bias=True), Linear(in_features=400, out_features=400, bias=True), Linear(in_features=400, out_features=400, bias=True), Linear(in_features=400, out_features=400, bias=True), Linear(in_features=400, out_features=400, bias=True)]
    
    
    
    """


    qf1_mask_generator = networks.MaskGeneratorNet(
        base_type=networks.MLPBase,
        traj_encoder_hidden_shape = params['traj_encoder']['hidden_shapes'],
        input_shape=env.observation_space.shape[0] * params['general_setting']["epoch_frames"],
        em_hidden_shapes=params['task_embedding']['em_hidden_shapes'],
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        trajectory_encoder=encoder,
        pruning_ratio=pruning_ratio,
        device="cpu")
    qf2_mask_generator = networks.MaskGeneratorNet(
        base_type=networks.MLPBase,
        traj_encoder_hidden_shape = params['traj_encoder']['hidden_shapes'],
        input_shape=env.observation_space.shape[0] * params['general_setting']["epoch_frames"],
        em_hidden_shapes=params['task_embedding']['em_hidden_shapes'],
        em_input_shape=np.prod(example_embedding.shape),
        num_layers=2, ##:
        hidden_shapes=params['net']['hidden_shapes'],
        trajectory_encoder=encoder,
        pruning_ratio=pruning_ratio,
        device="cpu")
    
    print("mask generator finish initialization")
    if args.qf1_snap is not None:
        qf1.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    if args.qf2_snap is not None:
        qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    
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


    # Mask buffer, stores the current masks for each layer, 
    # for each task and for each network type(Q1, Q2, policy).
    # Initialize the binary mask for all the weights and bias, make sure
    # follow the pruning ratio requirement.

    

    all_mask_buffer = {}

    # mask_buffer = {"Q1":{}, "Q2":{}, "Policy":{}}
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

            mask_buffer[task_idx] = neuron_masks
            #print("mask_buffer[net_type][each_task]",mask_buffer[net_type][each_task])

        all_mask_buffer[net_type] = mask_buffer       
        #print(id(all_mask_buffer[net_type]))
    #mask_buffer = mask_buffer.to(device)
    # Now we have a mask buffer like this:
    # {"Q1":{"0":[
    #                  [[layer1_Weights],[layer2_weights]..], 
    #                ],
    #
    # 
    # ,..."49":[]},"Q2":{...},"Policy":{...}}


    if RESTORE:
        with open(osp.join(osp.join(logger.work_dir,"model"), "replay_buffer.pkl"), 'rb') as f:
            replay_buffer = pickle.load(f)

    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['state_trajectory'] = state_trajectory
    #params['general_setting']['mask_buffer'] = mask_buffer
    #print("state_trajectory id",id(state_trajectory))
    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    


    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,state_trajectory=state_trajectory,
        mask_buffer=all_mask_buffer["Policy"],
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        manager=manager,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    """
            mask_generators = {"policy_mask_generator": policy_mask_generator,
                           "qf1_mask_generator": qf1_mask_generator,
                           "qf2_mask_generator": qf2_mask_generator},
    """

    agent = MUST_SAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        mask_generators = {"policy_mask_generator": policy_mask_generator,
                           "qf1_mask_generator": qf1_mask_generator,
                           "qf2_mask_generator": qf2_mask_generator},
        task_nums=env.num_tasks,
        mask_buffer=all_mask_buffer,
        traj_encoder=encoder,
        **params['sac'],
        **params['general_setting']
    )
    agent.train(env.num_tasks)


if __name__ == "__main__":
    experiment(args)
