
import torch
import copy
import numpy as np
import time

from .base import AsyncParallelCollector
import torch.multiprocessing as mp

import torchrl.policies as policies

from torchrl.env.get_env import *
from torchrl.env.continuous_wrapper import *

from metaworld_utils.meta_env import generate_single_mt_env

from metaworld_utils.meta_env import get_meta_env

from collections import OrderedDict


RESTORE = int(os.getenv('RESTORE', '0'))

class AsyncSingleTaskParallelCollector(AsyncParallelCollector):
    def __init__(
            self,
            reset_idx=False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(
            shared_pf, env_info, shared_que, start_barrier, epochs, reset_idx):

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)


        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())

            eval_rews = []

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):
                if reset_idx:
                    eval_ob = env_info.env.reset_with_index(idx)
                else:
                    eval_ob = env_info.env.reset()
                rew = 0
                current_success = 0
                while not done:
                    act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), )
                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()

                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
    
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.env_info.env_cls  = self.env_cls
        self.env_info.env_args = self.env_args

        for i in range(self.worker_nums):
            self.env_info.env_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, self.state_trajectory,
                    self.shared_que, self.start_barrier,
                    self.train_epochs))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        # self.eval_start_barrier.wait()
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
            mean_success_rate += worker_rst["success_rate"]

        return {
            'eval_rewards':eval_rews,
            'mean_success_rate': mean_success_rate / self.eval_worker_nums
        }



class AsyncMultiTaskParallelCollectorUniform(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1,**kwargs):
        super().__init__(**kwargs)

        self.progress_alpha = progress_alpha

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer, idx_mapping, neuron_masks, enable_mask):
        pf = funcs["pf"]
        ob = ob_info["ob"]

        pf.eval()
        time0 = time.time()

        with torch.no_grad():
            embedding_input = torch.zeros(env_info.num_tasks)
            embedding_input[env_info.env_rank] = 1
            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
            #print("env_info.device",env_info.device)
            out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
                                neuron_masks=neuron_masks,enable_mask=enable_mask)
            act = out["action"]


        act = act.detach().cpu().numpy()

        time1 = time.time()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = env_info.env.step(act)

        time2 = time.time()
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        time3 = time.time()


        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.env_rank],
            "rewards": [reward],
            "terminals": [done]
        }
        sample_dict["embedding_inputs"] = embedding_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample(sample_dict, env_info.env_rank)
        time4 = time.time()

        # print(f"time diff 0 {time1-time0}")
        # print(f"time diff 1 {time2-time1}")
        # print(f"time diff 2 {time3-time2}")
        # print(f"time diff 3 {time4-time3}")
        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict, mask_buffer, state_trajectory, 
        traj_collect_mod,
        index_mapping, lock, pretraining_epoch):

        # Attention: Here mask_buffer is the policy net weight masks for each task.
        # i.e. mask_buffer[task_id] = [all layer neuron masks]
        if not RESTORE:
            replay_buffer.rebuild_from_tag()

        # ALright, after deepcopy, the network has multiple local copies.
        local_funcs = copy.deepcopy(shared_funcs)

        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]
        mask_this_task = None

        if norm_obs_flag:
            shared_dict[task_name] = {
                "obs_mean": env_info.env._obs_mean,
                "obs_var": env_info.env._obs_var
            }
            # print("Put", task_name)

        ob = env_info.env.reset()
        c_ob = {
            "ob": ob
        }
        train_rew = 0
        current_epoch = 0

        while True:
            # For each episode:
            start_barrier.wait()

            # time to update local mask.
            #del mask_this_task
            mask_this_task = mask_buffer[env_info.env_rank]
            mask_this_task = [i.to(env_info.device) for i in copy.deepcopy(mask_this_task)]
            #mask_this_task = copy.deepcopy(mask_this_task).to(env_info.device)

            current_epoch += 1

            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            enable_mask = True
            if pretraining_epoch >= current_epoch:
                enable_mask = False

            for key in shared_funcs:
                # Load the base network's weights into this network copy.
                # Need to apply mask.
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_epoch_reward = 0    

            task_sample_index = shared_dict['task_sample_index']
 
            episode_state_traj = [ob]
            success = 0
            if env_info.env_rank in task_sample_index:
                
                for _ in range(env_info.epoch_frames):
                    next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer, index_mapping, mask_this_task, enable_mask)
                    c_ob["ob"] = next_ob
                    episode_state_traj.append(c_ob["ob"])
                    train_rew += reward
                    train_epoch_reward += reward

                    if done:

                        train_rews.append(train_rew)
                        train_rew = 0

                    if max(info["success"],success) > 0:
                        # this traj is success.
                        success = 1
                        
                        #print("############train success###########!")
                        #("episode_state_traj",episode_state_traj)
                # if len(episode_state_traj) != 201:
                #     print("len(episode_state_traj)",len(episode_state_traj),env_info.env_rank)
                #     assert 1==2
                
                # # if ever traj_collect_mod[env_info.env_rank] is 1, that means this task
                # # once succeed, we immediately switch the mode to only save successful traj states
                # # to state traj buffer.
                new_value = max(traj_collect_mod[env_info.env_rank],success)
                if traj_collect_mod[env_info.env_rank] == 0:
                    if new_value == 1:
                        # first time set this to 1:
                        # reset the task traj buffer, from now on the buffer shall only store
                        # successful traj.
                        state_trajectory[env_info.env_rank] = []

                        # set the new mod to 1.
                        traj_collect_mod[env_info.env_rank] = new_value
            
                if traj_collect_mod[env_info.env_rank] and success==0:
                    # Append the task state trajectory for this episode to the shared buffer.
                    print("first case")
                    print(next_ob)
                    pass

                else:
                    # Append the task state trajectory for this episode to the shared buffer.
                    #print("state_trajectory[env_info.env_rank] before",len(state_trajectory[env_info.env_rank]))

                    state_trajectory[env_info.env_rank] += [episode_state_traj]
                    #print("state_trajectory[env_info.env_rank] after",len(state_trajectory[env_info.env_rank]))
            

            if norm_obs_flag:
                shared_dict[task_name] = {
                    "obs_mean": env_info.env._obs_mean,
                    "obs_var": env_info.env._obs_var
                }
            del mask_this_task
            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf, 
                            env_info, 
                            shared_que, 
                            start_barrier,
                            epochs,
                            start_epoch,
                            task_name,
                            shared_dict,
                            mask_buffer,
                            state_trajectory,
                            traj_collect_mod,
                            lock
                            ):
        #TODO:
        # Attention, now your training and evaluation both collect trajectories,
        # this may cause write error for your buffer.

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        # print(env_info.env_args)

        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # Local mask for this task.
        mask_this_task = None
        

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            #del mask_this_task
            mask_this_task = mask_buffer[env_info.env_rank]
            mask_this_task = [i.to(env_info.device) for i in copy.deepcopy(mask_this_task)]
            current_epoch += 1

            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'success_rate': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break

            # Load the base network's weights into this network copy.
            # Need to apply mask.
            pf.load_state_dict(shared_pf.state_dict())

            # Here, apply the binary mask to the weight matrix.
            # Since weights are changing every episode, we need to do this 
            # every episode.
            #apply_mask(mask_this_task, pf)

            # switch mode to evaluation.
            pf.eval()

            if norm_obs_flag:
                env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                # print(env_info.env._obs_mean)
                #  = {
                #     "obs_mean": env_info.env._obs_mean,
                #     "obs_var": env_info.env._obs_var
                # }

            eval_rews = []  

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                current_success = 0
                episode_state_traj = [eval_ob]
                while not done:


                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), mask_this_task)

                    eval_ob, r, done, info = env_info.env.step( act )
                    episode_state_traj.append(eval_ob)
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    current_success = max(current_success, info["success"])

                # Append the task state trajectory for this episode to the shared buffer.
                #state_trajectory[env_info.env_rank] += [episode_state_traj]

                eval_rews.append(rew)
                done = False
                success += current_success


                # if ever traj_collect_mod[env_info.env_rank] is 1, that means this task
                # once succeed, we immediately switch the mode to only save successful traj states
                # to state traj buffer.
                new_value = 0
                if current_success > 0:
                    new_value = 1

                if traj_collect_mod[env_info.env_rank] == 0:
                    if new_value == 1:
                        # first time set this to 1:
                        # reset the task traj buffer, from now on the buffer shall only store
                        # successful traj.
                        state_trajectory[env_info.env_rank] = []

                        # set the new mod to 1.
                        traj_collect_mod[env_info.env_rank] = new_value
                
                if traj_collect_mod[env_info.env_rank] and new_value==0:
                    # Append the task state trajectory for this episode to the shared buffer.
                    pass

                else:
                    # Append the task state trajectory for this episode to the shared buffer.
                    #print("state_trajectory[env_info.env_rank] before",len(state_trajectory[env_info.env_rank]))

                    state_trajectory[env_info.env_rank] += [episode_state_traj]
                    #print("state_trajectory[env_info.env_rank] after",len(state_trajectory[env_info.env_rank]))
            del mask_this_task
            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes,
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


        self.shared_dict = self.manager.dict()
        self.shared_dict['task_sample_index'] = list(range(self.task_amount))  #*

        #print("self.mask_buffer",id(self.mask_buffer))

        assert self.worker_nums == self.env.num_tasks
        # task_cls, task_args, env_params
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }
        
        tasks = list(self.env_cls.keys())
        lock = self.manager.Lock()
        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]
            
            self.env_info.env_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            if "start_epoch" in self.env_info.env_args["task_args"]:
                start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            else:
                start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            #print("state_trajectory id in start_worker",id(self.state_trajectory))
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer,
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, 
                    self.shared_dict, 
                    self.mask_buffer, 
                    self.state_trajectory, 
                    self.traj_collect_mod,
                    self.index_mapping,lock,self.pretraining_epoch))  #*
            p.start()
            self.workers.append(p)
            


        assert self.eval_worker_nums == self.env.num_tasks
        
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0
            """
            shared_pf, 
                            env_info, 
                            shared_que, 
                            start_barrier,
                            epochs,
                            start_epoch,
                            task_name,
                            shared_dict,
                            mask_buffer,
                            state_trajectory
            
            """

            self.env_info.env_args["env_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                      self.env_info, 
                      self.eval_shared_que, 
                      self.eval_start_barrier,
                      self.eval_epochs, 
                      start_epoch, 
                      task, 
                      self.shared_dict,
                      self.mask_buffer, 
                      self.state_trajectory,
                      self.traj_collect_mod,
                      lock
                      ))  #*
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                mean_success_rate += worker_rst["success_rate"]
                tasks_result.append((worker_rst["task_name"], worker_rst["success_rate"], np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, success_rate, eval_rewards in tasks_result:
            dic[task_name+"_success_rate"] = success_rate
            dic[task_name+"_eval_rewards"] = eval_rewards
            dic[str(self.tasks_mapping[task_name])] = success_rate
            # if success_rate > 0:
            #     print("kkk task_name success",task_name)
            # if self.tasks_progress[self.tasks_mapping[task_name]] is None:
            #     self.tasks_progress[self.tasks_mapping[task_name]] = success_rate
            # else:
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)
            self.tasks_progress[self.tasks_mapping[task_name]] += \
                self.progress_alpha * success_rate

        dic['eval_rewards']      = eval_rews
        dic['mean_success_rate'] = mean_success_rate / active_task_counts

        return dic


    def train_one_epoch(self, task_sample_index):
        train_rews = []
        train_epoch_reward = 0

        # Base network load weight.
        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())

        self.shared_dict['task_sample_index'] = task_sample_index

        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            if worker_rst["train_rewards"] is not None:
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums

        print(f'replay_buffer._size: {self.replay_buffer._size}')

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }


