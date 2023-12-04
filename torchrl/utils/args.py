import argparse
import json

import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--pruning_ratio', type=float, default=0.7,
                        help='prining ratio between 0 and 1.')
    parser.add_argument('--generator_lr', type=float, default=1e-4,
                        help='generator sl loss lr.')
    parser.add_argument('--use_sl_loss', type=int, default=1,
                        help='if 1, enable sl loss.')
    parser.add_argument('--selected_task_amount', type=int, default=1,
                        help='only use a few tasks from metaworld.')

    parser.add_argument('--use_trajectory_info', type=int, default=1,
                        help='if true, also uses trajectory info for generator')
    
    parser.add_argument('--wandb_group_name', type=str, default="default_group",
                        help='wandb group name')

    parser.add_argument('--success_traj_update_only', type=int, default=1,
                        help='If enabled, only the tasks with success trajs, and success rate < 0.66 will be updated')

    parser.add_argument('--mask_update_interval', type=int, default=25,
                        help='mask update interval')

    parser.add_argument('--sl_optim_times', type=int, default=5,
                        help='mask update interval')
    
    parser.add_argument('--mask_end_update_episode', type=int, default=10000,
                        help='mask end update')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')

    parser.add_argument('--worker_nums', type=int, default=4,
                        help='worker nums')

    parser.add_argument('--eval_worker_nums', type=int, default=2,
                        help='eval worker nums')

    parser.add_argument("--config", type=str,   default=None,
                        help="config file", )

    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='directory for snapshots (default: ./snapshots)')

    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification", )

    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )

    # policy snapshot
    parser.add_argument("--pf_snap", type=str,   default=None,
                        help="policy snapshot path", )
    # q function snapshot
    parser.add_argument("--qf1_snap", type=str,   default=None,
                        help="policy snapshot path", )
    # q function snapshot
    parser.add_argument("--qf2_snap", type=str,   default=None,
                        help="policy snapshot path", )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params
