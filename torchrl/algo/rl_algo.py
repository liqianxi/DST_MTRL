import copy
import pickle, wandb
import time
from collections import deque
import numpy as np

import torch, random

import torchrl.algo.utils as atu
import torch.nn.functional as F
import gym

import os
import os.path as osp
#import wandb

from torchrl.task_scheduler import TaskScheduler
import torchrl.plot_history as plot_history
from tqdm import tqdm
#from torch.profiler import profile, record_function, ProfilerActivity

TASK_SAMPLE_NUM = int(os.getenv('TASK_SAMPLE_NUM', '50'))
RESTORE = int(os.getenv('RESTORE', '0'))
ID = os.getenv('ID', 'null')
DIR = os.getenv('DIR', 'null')
EPOCH = int(os.getenv('EPOCH', '0'))




class RLAlgo():
    """
    Base RL Algorithm Framework
    """

    def __init__(self,
                 env=None,
                 replay_buffer=None,
                 collector=None,
                 logger=None,
                 continuous=None,
                 discount=0.99,
                 num_epochs=3000,
                 epoch_frames=1000,
                 max_episode_frames=999,
                 batch_size=128,
                 device='cpu',
                 train_render=False,
                 eval_episodes=1,
                 eval_render=False,
                 save_interval=100,
                 save_dir=None,
                 mask_buffer=None,
                 mask_generators=None,
                 mask_update_interval=None,
                 state_trajectory=None,
                 update_end_epoch=5000,
                 trajectory_encoder=None,
                 traj_collect_mod=None,
                 generator_lr=1e-5,
                 recent_traj_window=20,
                 success_traj_update_only=True,
                 final_mask=None
                 ):

        self.env = env
        self.final_mask=final_mask
        self.success_traj_update_only = success_traj_update_only
        #print("self.success_traj_update_only",self.success_traj_update_only)
        self.mask_update_interval = mask_update_interval
        self.state_trajectory = state_trajectory
        self.policy_mask_generator = mask_generators["policy_mask_generator"]
        self.qf1_mask_generator = mask_generators["qf1_mask_generator"]
        self.qf2_mask_generator = mask_generators["qf2_mask_generator"]
        self.recent_traj_window = recent_traj_window
        self.traj_collect_mod=traj_collect_mod

        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.traj_encoder = trajectory_encoder
        self.replay_buffer = replay_buffer
        self.collector = collector
        # device specification
        self.device = device

        self.mask_generator_optimizer = torch.optim.Adam([
            {'params':self.policy_mask_generator.parameters()},
            {'params':self.qf1_mask_generator.parameters()},
            {'params':self.qf2_mask_generator.parameters()}
        ], lr=generator_lr)


        self.mask_buffer = mask_buffer

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames
        self.update_end_epoch = update_end_epoch

        self.train_render = train_render
        self.eval_render = eval_render

        # training information
        self.batch_size = batch_size
        self.training_update_num = 0
        self.sample_key = None


        # Logger & relevant setting
        self.logger = logger

        self.episode_rewards = deque(maxlen=30)
        self.training_episode_rewards = deque(maxlen=30)
        self.eval_episodes = eval_episodes

        self.save_interval = save_interval
        self.save_dir = save_dir
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.best_eval = None

        self.success_rate_dict = {}

        for i in range(10):
            self.success_rate_dict[i] = deque(maxlen=3)



    def construct_one_hot_map(self, total_tasks):
        one_hot_map = {}
        for i in range(total_tasks):
            embedding_input = torch.zeros(total_tasks)
            embedding_input[i] = 1
            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
            embedding_input = embedding_input.unsqueeze(0).to(self.device)
            one_hot_map[i] = embedding_input

        return one_hot_map
    
    def euclidean_distance_matrix(self,tensors):

        # Stack the 1D tensors to create a 2D tensor
        stacked_tensors = torch.stack(tensors)

        # Calculate the pairwise squared Euclidean distances
        squared_distances = torch.cdist(stacked_tensors,stacked_tensors)

        return squared_distances

    def start_epoch(self):
        pass

    def finish_epoch(self):
        return {}

    def pretrain(self):
        pass

    def update_per_epoch(self):
        pass

    def snapshot(self, prefix, epoch):
        print(f'snapshot at {epoch}')

        for name, network in self.snapshot_networks:
            model_file_name = "model_{}_{}.pth".format(name, epoch)
            model_path = osp.join(prefix, model_file_name)
            torch.save(network.state_dict(), model_path)
        
        torch.save(self.log_alpha, osp.join(prefix, "log_alpha_{}.pth".format(epoch)))

        with open(osp.join(prefix, "replay_buffer.pkl"), 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        

    def get_traj_availability(self,state_trajectory):
        key_list = []
        for key,value in state_trajectory.items():
            if len(value) > 0:
                key_list.append(key)

        return key_list

    def clip_by_window(self,list_of_trajs,window_length):
        if len(list_of_trajs) > window_length:
            return list_of_trajs[-window_length:]
        return list_of_trajs

    def encode_policy_into_vectors(self, network, trajectories):
        """
        Encode a policy, represented by sampled trajectories, into a single diagonal Gaussian
        by embedding trajectories and fitting a Gaussian distribution on the latents.

        Returns th.distributions.MultivariateNormal
        """
        #print("trajectories",trajectories.shape)
        latents = network.encode(trajectories)

        return latents

    def compute_mask_loss(self,traj_sim_mtx, mask_sim_mtx):
        # epsilon = 1e-8
        # tensor1 = mask_sim_mtx / mask_sim_mtx.sum() + epsilon
        # tensor2 = traj_sim_mtx / traj_sim_mtx.sum() + epsilon

        #torch.reshape(torch.sum(tensor1 * torch.log(tensor1 / tensor2)), (1, 1)), 
        return torch.cdist(traj_sim_mtx, mask_sim_mtx)

    def compute_policy_similarity_matrix(self, task_amount, task_traj_batch):
        #recent_trajs: task_amount*traj_size*
        with torch.no_grad():
            encoding = self.encode_policy_into_vectors(self.traj_encoder, 
                                            task_traj_batch)

            list_encoding = [i for i in encoding]

            distance_matrix = self.euclidean_distance_matrix(list_encoding)

            similarity_matrix = torch.exp(-distance_matrix)

            max_value = torch.max(similarity_matrix)
            min_value = torch.min(similarity_matrix)
            similarity_matrix = (similarity_matrix - min_value) / (max_value - min_value)

            return similarity_matrix.reshape(1,task_amount*task_amount)

    def compute_mask_similarity_matrix(self, mask_buffer, task_amount):
        task_mask_list = mask_buffer

        distances = self.euclidean_distance_matrix(task_mask_list)

        similarities = torch.exp(-distances)
        max_value = torch.max(similarities)
        min_value = torch.min(similarities)
        similarities = (similarities - min_value) / (max_value - min_value)

        return similarities.reshape(1,task_amount*task_amount)

    def check_finish_update(self,success_rate_dict):
        #print("success_rate_dict",success_rate_dict)
        avg = sum(success_rate_dict) / len(success_rate_dict)
        return avg >= 0.66

    def sample_update_data(self,device):
        #self.success_rate_dict
        #self.state_trajectory
        update_batch = []
        #update_task_list = []
        
        for task_id, task_mod in self.traj_collect_mod.items():
            task_state_traj_buffer = self.state_trajectory[task_id]
            #print("task_state_traj_buffer",task_state_traj_buffer)
            if task_mod == 1:
                traj = random.choice(task_state_traj_buffer)

            else: 
                traj = task_state_traj_buffer[-1]

            update_batch.append(torch.as_tensor(traj))
            #print(update_batch[-1].shape)

        return torch.stack(update_batch).float().to(device)

    
    def update_masks(self, sampled_task_amount,all_task_amount, current_epoch):
        # First, update encoder

        recent_window = self.recent_traj_window   
        for t_id in range(all_task_amount):
            self.state_trajectory[t_id] = self.clip_by_window(self.state_trajectory[t_id],recent_window)    

        for each_net in ["Policy","Q1","Q2"]:



            mask_sim_mtx, traj_sim_mtx = None, None
            batch_task_probs_masks, batch_task_binary_masks = None, None
            loss = None

            for optim_time in range(5):
                task_traj_batch = self.sample_update_data(self.device)
                #print("task_traj_batch",task_traj_batch.shape)task_traj_batch torch.Size([10, 19])

                task_onehot_batch = torch.stack([self.one_hot_map[i].squeeze(0) for i in range(all_task_amount)]).to(self.device)

                generator = self.policy_mask_generator
                if each_net == "Q1":
                    generator = self.qf1_mask_generator
                elif each_net == "Q2":
                    generator = self.qf2_mask_generator

                batch_task_probs_masks, batch_task_binary_masks = generator(task_traj_batch, task_onehot_batch)

                # Calculate two losses.
                mask_sim_mtx = self.compute_mask_similarity_matrix(batch_task_probs_masks, all_task_amount)
                traj_sim_mtx = self.compute_policy_similarity_matrix(all_task_amount, task_traj_batch)
                loss= self.compute_mask_loss(traj_sim_mtx, mask_sim_mtx)
                self.mask_generator_optimizer.zero_grad()
            
                norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
                # the mask generator network will be updated.
                loss.backward()
                self.mask_generator_optimizer.step()

            wandb.log({f"{each_net}_mask_sim_mtx":mask_sim_mtx},step=current_epoch)
            wandb.log({f"{each_net}_traj_sim_mtx":traj_sim_mtx},step=current_epoch)
            tmp_dict = {}
            for each_task in range(sampled_task_amount):
                
                if self.success_traj_update_only:
                    if self.traj_collect_mod[each_task] and not self.check_finish_update(self.success_rate_dict[each_task]):
                        #self.mask_buffer[each_net][each_task] = [i.clone().detach() for i in batch_task_binary_masks[each_task]]
                        tmp_dict[each_task] = [i.clone().detach() for i in batch_task_binary_masks[each_task]]
                else: 
                    tmp_dict[each_task] = [i.clone().detach() for i in batch_task_binary_masks[each_task]]

                    #self.mask_buffer[each_net][each_task] = [i.clone().detach() for i in batch_task_binary_masks[each_task]]

            self.mask_buffer[each_net].update(tmp_dict)
            #print("loss1",loss1)
            #print("loss2",loss2)
            wandb.log({f"{each_net}_sim_loss":loss},step=current_epoch)
            #wandb.log({f"{each_net}_sim_loss2":loss2},step=current_epoch)

            


    def mask_update_scheduler(self, method, epoch, update_end_epoch,freq=None):
        assert update_end_epoch != None
        if method == "fix_interval":

            if epoch < update_end_epoch and epoch !=0:
                assert freq != None
                return epoch % freq == 0
            else: 
                return False

    def train(self, task_amount,params, group_name):
        global EPOCH
        self.all_task_amount = task_amount
        assert task_amount in [10,50]
        self.one_hot_map = self.construct_one_hot_map(task_amount)
        
        wandb.init(
            project="dst_mtrl",
            group=group_name,
            settings=wandb.Settings(start_method="fork"),
            config=params
            )

        #wandb.watch((self.policy_mask_generator,self.qf1_mask_generator,self.qf2_mask_generator),log='all',log_freq=5,log_graph=True)

        if RESTORE:
            for name, network in self.snapshot_networks:
                model_file_name = "model_{}_{}.pth".format(name, EPOCH)
                model_path = osp.join(self.save_dir, model_file_name)
                network.load_state_dict(torch.load(model_path))

            self.log_alpha = torch.load(osp.join(self.save_dir, "log_alpha_{}.pth".format(EPOCH)))

            EPOCH += 1

        self.pretrain(task_amount)

        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames

        #*
        self.start_epoch()
        task_scheduler = TaskScheduler(num_tasks=task_amount, task_sample_num=TASK_SAMPLE_NUM)

        # For each episode:
        for epoch in tqdm(range(EPOCH, self.num_epochs)):
            wandb.log({"save_traj_mod_sum":sum([i for i in self.traj_collect_mod.values()])},step=epoch)
            
            torch.cuda.empty_cache()
            for each_task in self.mask_buffer["Policy"].keys():
                value = torch.sum((self.mask_buffer["Policy"][each_task][0] == 0).nonzero().squeeze()).item()

                name = str(each_task)
                wandb.log({f"task_policy_mask_{name}":value},step=epoch)
            #print("self.update_end_epoch",self.update_end_epoch)
            if self.mask_update_scheduler("fix_interval", epoch, self.update_end_epoch,freq=self.mask_update_interval):
                # update mask
                print("start to update mask")
                print(self.success_rate_dict)
                self.update_masks(TASK_SAMPLE_NUM, task_amount, epoch)

            if epoch >= self.update_end_epoch:
                for net_type in ["Q1","Q2","Policy"]:
                    for task_each_id in range(task_amount):
                        self.final_mask[net_type][task_each_id] = copy.deepcopy(self.mask_buffer[net_type][task_each_id])
                self.mask_buffer = self.final_mask
                #print("new final mask buffer",self.mask_buffer)

            log_dict = {}

            self.current_epoch = epoch
            start = time.time()
            # If only a subset of task is sampled:
            for _ in range(task_scheduler.num_tasks // TASK_SAMPLE_NUM):
                task_sample_index = task_scheduler.sample()

                self.start_epoch()

                explore_start_time = time.time()
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                training_epoch_info = self.collector.train_one_epoch(
                    task_sample_index)

                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                for reward in training_epoch_info["train_rewards"]:
                    self.training_episode_rewards.append(reward)
                explore_time = time.time() - explore_start_time

                train_start_time = time.time()

            torch.cuda.empty_cache()
            self.update_per_epoch(task_sample_index, task_scheduler, self.mask_buffer, epoch)

            train_time = time.time() - train_start_time
            print("train_time",train_time)
            finish_epoch_info = self.finish_epoch()

            eval_start_time = time.time()
            eval_infos = self.collector.eval_one_epoch()
            #print("eval_infos")

            for task_id in range(task_scheduler.num_tasks):
                self.success_rate_dict[task_id].append(eval_infos[str(task_id)])

            # task_scheduler.update_success_rate_array(eval_infos)
            # task_scheduler.update_return_array(eval_infos)
            # task_scheduler.update_p()

            eval_time = time.time() - eval_start_time
            #print("eval time",eval_time)
            total_frames += self.collector.active_worker_nums * self.epoch_frames

            infos = {}

            for reward in eval_infos["eval_rewards"]:
                self.episode_rewards.append(reward)
            # del eval_infos["eval_rewards"]

            if self.best_eval is None or \
                    np.mean(eval_infos["eval_rewards"]) > self.best_eval:
                self.best_eval = np.mean(eval_infos["eval_rewards"])
                self.snapshot(self.save_dir, 'best')
            del eval_infos["eval_rewards"]

            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(
                self.training_episode_rewards)
            infos["Explore_Time"] = explore_time
            infos["Train___Time"] = train_time
            infos["Eval____Time"] = eval_time
            infos.update(eval_infos)
            infos.update(finish_epoch_info)
            wandb.log(infos,step=epoch)

            log_dict['mean_success_rate'] = infos['mean_success_rate']

            self.logger.add_epoch_info(epoch, total_frames,
                                       time.time() - start, infos)

            if epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, epoch)

                task_scheduler.save(self.save_dir)
                # filename = plot_history.plot(root_dir=self.save_dir, num_tasks=task_scheduler.num_tasks, sample_gap=1, perfermance_gap=10, delta_gap=10)
                # log_dict['history'] = wandb.Image(filename)

            wandb.log(log_dict)

        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()

    def update(self, batch):
        raise NotImplementedError

    def _update_target_networks(self):
        if self.use_soft_update:
            for net, target_net in self.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.target_networks:
                    atu.copy_model_params_from_to(net, target_net)

    @property
    def networks(self):
        return [
        ]

    @property
    def snapshot_networks(self):
        return [
        ]

    @property
    def target_networks(self):
        return [
        ]

    def to(self, device):
        for net in self.networks:
            net.to(device)
