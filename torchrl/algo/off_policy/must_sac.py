from matplotlib.pyplot import axis
from nupic.torch.modules import rezero_weights
from .twin_sac_q import TwinSACQ
import copy
import torch
import numpy as np

import torchrl.policies as policies
import torch.nn.functional as F


class MUST_SAC(TwinSACQ):
    """"
    Support Different Temperature for different tasks
    """
    def __init__(self, task_nums,
                 temp_reweight=False,
                 grad_clip=True,
                 **kwargs):

        super().__init__(**kwargs)

        self.task_nums = task_nums
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(self.task_nums).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )
        self.sample_key = ["obs", "next_obs", "acts", "rewards",
                           "terminals",  "task_idxs"]

        self.pf_flag = isinstance(self.pf,
                                  policies.EmbeddingGuassianContPolicyBase)

        self.idx_flag = isinstance(self.pf, policies.MultiHeadGuassianContPolicy)

        self.temp_reweight = temp_reweight

        self.sample_key.append("embedding_inputs")
        self.grad_clip = grad_clip

    def update(self, batch, task_sample_index, task_scheduler, mask_buffer):
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)

        batch_size = batch['obs'].shape[0]
        each_task_batch_size = 0
        if task_scheduler.task_sample_num == 10:
            update_idxes = np.ones(10, dtype=np.int32) * (batch_size // 10)
            each_task_batch_size = batch_size // 10
        elif task_scheduler.task_sample_num == 50:
            update_idxes = np.ones(50, dtype=np.int32) * (batch_size // 50)
            each_task_batch_size = batch_size // 50
        else:
            update_idxes = (task_scheduler.p * batch_size).astype(np.int32)
            update_idxes[-1] = batch_size - np.sum(update_idxes[:-1])

        # print(f'update_idxes: {update_idxes}')

        obs = torch.cat([obs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        actions = torch.cat([actions[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        next_obs = torch.cat([next_obs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        rewards = torch.cat([rewards[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        terminals = torch.cat([terminals[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])

        embedding_inputs = batch["embedding_inputs"]
        embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)
        embedding_inputs = torch.cat([embedding_inputs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])


        task_idx = batch['task_idxs']
        task_idx = torch.Tensor(task_idx).to( self.device ).long()
        task_idx = torch.cat([task_idx[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        #print("task_idx",task_idx.shape) #task_idx torch.Size([1280, 1])
        # Here task idx: 0000000... 111111.. ...999999
        # if self.idx_flag:
        #     task_idx    = torch.Tensor(task_idx).to( self.device ).long()

        self.pf.train()
        self.qf1.train()
        self.qf2.train()

        """
        Policy operations.
        """
        # 10*128

        task_idx = torch.reshape(task_idx, (task_scheduler.task_sample_num,
                                            each_task_batch_size))

        # RuntimeError: shape '[10, 128]' is invalid for input of size 12800
        # Reshape all the replay buffer data as 10*128*-1

        embedding_inputs = torch.reshape(embedding_inputs, (task_scheduler.task_sample_num,
                                            each_task_batch_size, -1))

        obs = torch.reshape(obs, (task_scheduler.task_sample_num,
                                            each_task_batch_size, -1))

        actions = torch.reshape(actions, (task_scheduler.task_sample_num,
                                            each_task_batch_size, -1))

        next_obs = torch.reshape(next_obs, (task_scheduler.task_sample_num,
                                            each_task_batch_size, -1))
        #print("rewards",rewards.shape) #rewards torch.Size([1280, 1])
        rewards = torch.reshape(rewards, (task_scheduler.task_sample_num,
                                            each_task_batch_size, 1))
        #print("terminals",terminals.shape)
        terminals = torch.reshape(terminals, (task_scheduler.task_sample_num,
                                            each_task_batch_size, 1))
        
        # First, let's make sure several things:
        # 1. We have, say policy net, Q1 net and Q2 net, in total 3 base networks,
        # and for each net, for each of the 10/50 tasks, we need to do a batch update using different masks.
        # so the follow code must be organized in this way:
        #
        # for each_task_batch:
        #       run batch policy net with the mask - mask[policy][task]
        #       run batch q1 net with the mask - mask[q1][task]
        #       run batch q2 net with the mask - mask[q1][task]
        
        all_info = []
        for each_task_batch_idx in range(task_idx.shape[0]):
            mean = []
            log_std = []
            new_actions = []
            log_probs = []
            # 1*128
            one_task_idx = task_idx[each_task_batch_idx][0] # Get the task id.
            policy_mask_layers = mask_buffer["Policy"][one_task_idx.item()]
            policy_device_masks = [i.to(self.device) for i in policy_mask_layers]
            
            sample_info = self.pf.explore(obs[each_task_batch_idx], 
                                          neuron_masks=policy_device_masks,
                                          return_log_probs=True)

            mean = sample_info["mean"] #128*xx
            log_std = sample_info["log_std"]
            new_actions = sample_info["action"]
            log_probs = sample_info["log_prob"]

            # Here, only take the 
            cat_input = torch.cat([obs[each_task_batch_idx], 
                                   actions[each_task_batch_idx]],
                                   dim=1)

            q1_device_masks = [i.to(self.device) for i in mask_buffer["Q1"][one_task_idx.item()]]
            q2_device_masks = [i.to(self.device) for i in mask_buffer["Q2"][one_task_idx.item()]]
            q1_pred = self.qf1(cat_input,q1_device_masks)
            q2_pred = self.qf2(cat_input,q2_device_masks)

        

            # reweight_coeff = 1
            reweight_coeff = torch.ones((log_probs.shape[0], 1)).to(self.device)

            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                batch_size = log_probs.shape[0]
                log_alpha = self.log_alpha[each_task_batch_idx].expand(batch_size, 1)

                #log_alphas = torch.unsqueeze(self.log_alpha, 0).expand(batch_size, -1)

                # if self.pf_flag:
                #     task_ids = torch.where(embedding_inputs[each_task_batch_idx] == 1)[1]
                # else:
                #     task_ids = torch.where(obs[:, -task_scheduler.num_tasks:] == 1)[1]
                
                # log_alphas = torch.gather(self.log_alpha, 0, task_ids).unsqueeze(-1)
                #print("log_alpha shape",log_alpha.shape)
                #print("log_probs shape",log_probs.shape)
                alpha_loss = -(log_alpha *
                            (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                #alphas = torch.gather(self.log_alpha.exp(), 0, each_task_batch_idx).unsqueeze(-1)
                alphas = log_alpha.exp()

                # if self.temp_reweight:
                #     softmax_temp = F.softmax(-log_alpha.detach())
                #     reweight_coeff = torch.gather(softmax_temp, 0, task_ids).unsqueeze(-1) * self.task_nums
            else:
                alphas = 1
                alpha_loss = 0

            with torch.no_grad():

                target_sample_info = self.pf.explore(next_obs[each_task_batch_idx],
                                                        neuron_masks=policy_device_masks,
                                                        return_log_probs=True)

                target_actions = target_sample_info["action"]
                target_log_probs = target_sample_info["log_prob"]

                cat_input = torch.cat([next_obs[each_task_batch_idx], 
                                                target_actions], dim=1)

                target_q1_pred = self.target_qf1(cat_input,
                                                  q1_device_masks
                                                )
                target_q2_pred = self.target_qf2(cat_input,
                                                  q2_device_masks
                                                )

                min_target_q = torch.min(target_q1_pred, target_q2_pred)
                #print("min_target_q",min_target_q.shape)
                target_v_values = min_target_q - alphas[:, :] * target_log_probs
            #print("self.discount",self.discount)
            """
            QF Loss
            """
            # q_target = rewards + (1. - terminals) * self.discount * target_v_values
            # There is no actual terminate in meta-world -> just filter all time_limit terminal
            # print("target_v_values,shape",target_v_values.shape)
            # print("rewards,shape",rewards[each_task_batch_idx].shape)
            
            q_target = rewards[each_task_batch_idx] + self.discount * target_v_values
            #print("q_target",q_target.shape)
            qf1_loss = (reweight_coeff[:, :] *
                        ((q1_pred - q_target.detach()) ** 2)).mean()
            qf2_loss = (reweight_coeff[:, :] *
                        ((q2_pred - q_target.detach()) ** 2)).mean()

            assert q1_pred.shape == q_target.shape
            assert q2_pred.shape == q_target.shape



            cat_input = torch.cat([obs[each_task_batch_idx], 
                                   new_actions],
                                   dim=1)
            q_new_actions = torch.min(
                self.qf1(cat_input,q1_device_masks),
                self.qf1(cat_input,q1_device_masks))

            """
            Policy Loss
            """
            if not self.reparameterization:
                raise NotImplementedError
            else:
                assert log_probs.shape == q_new_actions.shape
                policy_loss = (reweight_coeff[:, :] *
                            (alphas[:, :] * log_probs - q_new_actions)).mean()

            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

            policy_loss += std_reg_loss + mean_reg_loss

            """
            Update Networks
            """

            self.pf_optimizer.zero_grad()
            policy_loss.backward()
            if self.grad_clip:
                pf_norm = torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 1)
            self.pf_optimizer.step()
            self.pf.apply(rezero_weights)

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            if self.grad_clip:
                qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 1)
            self.qf1_optimizer.step()
            self.qf1.apply(rezero_weights)

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            if self.grad_clip:
                qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 1)
            self.qf2_optimizer.step()
            self.qf2.apply(rezero_weights)

            self._update_target_networks()

            info = {}

            # Information For Logger
            

            info['Reward_Mean'] = rewards.mean().item()

            if self.automatic_entropy_tuning:
                for i in range(self.task_nums):
                    info["alpha_{}".format(i)] = self.log_alpha[i].exp().item()
                info["Alpha_loss"] = alpha_loss.item()
            info['Training/policy_loss'] = policy_loss.item()
            info['Training/qf1_loss'] = qf1_loss.item()
            info['Training/qf2_loss'] = qf2_loss.item()

            if self.grad_clip:
                info['Training/pf_norm'] = pf_norm.item()
                info['Training/qf1_norm'] = qf1_norm.item()
                info['Training/qf2_norm'] = qf2_norm.item()

            info['log_std/mean'] = log_std.mean().item()
            info['log_std/std'] = log_std.std().item()
            info['log_std/max'] = log_std.max().item()
            info['log_std/min'] = log_std.min().item()

            # log_probs_display = log_probs.detach()
            # log_probs_display = (log_probs_display.mean(0)).squeeze(1)
            # for i in range(self.task_nums):
            #     info["log_prob_{}".format(i)] = log_probs_display[i].item()

            info['log_probs/mean'] = log_probs.mean().item()
            info['log_probs/std'] = log_probs.std().item()
            info['log_probs/max'] = log_probs.max().item()
            info['log_probs/min'] = log_probs.min().item()

            info['mean/mean'] = mean.mean().item()
            info['mean/std'] = mean.std().item()
            info['mean/max'] = mean.max().item()
            info['mean/min'] = mean.min().item()

            all_info.append(info)
            

            

        return all_info

    def update_per_epoch(self, task_sample_index, task_scheduler, mask_buffer):
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    self.task_nums,
                                                    task_sample_index=task_sample_index,
                                                    reshape=False)
            # Here, mask_buffer is all network types and all tasks.
            all_info = self.update(batch, task_sample_index, task_scheduler, mask_buffer)
            for each in all_info:
                self.logger.add_update_info(each)
        
        # print(f'num_steps_can_sample: {self.replay_buffer.num_steps_can_sample()}')
