from matplotlib.pyplot import axis
from nupic.torch.modules import rezero_weights
from .twin_sac_q import TwinSACQ
import copy
import torch
import numpy as np
import wandb,time
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

    # def concat_mask_tensors(self, sample_task_amount, specific_mask_buffer, task_batch_size, device):
    #     mask_layers = None
    #     for i in range(sample_task_amount):
    #         single_mask = [each.expand(task_batch_size, -1).to(device) for each in specific_mask_buffer[i]]
            
    #         if mask_layers:
    #             for j in range(len(mask_layers)):
    #                 mask_layers[j] = torch.cat((mask_layers[j], single_mask[j]), 0)
    #         else: 
    #             mask_layers = single_mask

    #     return mask_layers
    def concat_mask_tensors(self, sample_task_amount, specific_mask_buffer, task_batch_size, device):
        mask_layers = None
        for i in range(sample_task_amount):
            single_mask = [each.expand(task_batch_size, -1).to(device) for each in specific_mask_buffer[i]]
            
            if mask_layers is None:
                mask_layers = torch.empty((0,) + single_mask[0].shape[1:], device=device)

            for j in range(len(single_mask)):
                mask_layers = torch.cat((mask_layers, single_mask[j]), dim=0)

        return mask_layers

    def get_batch_size_and_idx(self,task_scheduler):
        batch_size = 1280
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

        return each_task_batch_size, update_idxes

    def update(self, batch, task_sample_index, task_scheduler, mask_buffer,
               each_task_batch_size, update_idxes,
               policy_device_masks,q1_device_masks,q2_device_masks
               ):
        self.training_update_num += 1
        time00 = time.time()
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

        obs = torch.cat([obs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        actions = torch.cat([actions[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        next_obs = torch.cat([next_obs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        rewards = torch.cat([rewards[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        terminals = torch.cat([terminals[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        #print("obs.device",obs.device)
        embedding_inputs = batch["embedding_inputs"]

        embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)
        embedding_inputs = torch.cat([embedding_inputs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        time000 = time.time()

        task_idx = batch['task_idxs']
        task_idx = torch.Tensor(task_idx).to( self.device ).long()
        task_idx = torch.cat([task_idx[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])
        #print("task_idx",task_idx.shape) #task_idx torch.Size([1280, 1])
        # Here task idx: 0000000... 111111.. ...999999

        self.pf.train()
        self.qf1.train()
        self.qf2.train()

        """
        Policy operations.
        """


        # (1280,xx)
        # 1*128
        time1 = time.time()

        #print("policy_device_masks.device",policy_device_masks[0].device)
        time2 = time.time()
        sample_info = self.pf.explore(obs, 
                                      neuron_masks=policy_device_masks,
                                      return_log_probs=True)

        time3 = time.time()
        mean = sample_info["mean"] #1280*xx
        log_std = sample_info["log_std"]
        new_actions = sample_info["action"]
        log_probs = sample_info["log_prob"]


        cat_input = torch.cat([obs, actions], dim=1)

        q1_pred = self.qf1(cat_input, q1_device_masks)
        q2_pred = self.qf2(cat_input, q2_device_masks)
        time4 = time.time()
        # reweight_coeff = 1
        reweight_coeff = torch.ones((log_probs.shape[0], 1)).to(self.device)

        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            task_ids = torch.where(embedding_inputs == 1)[1]
            batch_size = log_probs.shape[0]
            log_alphas = torch.unsqueeze(self.log_alpha, 0).expand(batch_size, -1)
            log_alphas = torch.gather(self.log_alpha, 0, task_ids).unsqueeze(-1)

            alpha_loss = -(log_alphas[:,:] *
                        (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            time5 = time.time()

            alphas = torch.gather(self.log_alpha.exp(), 0, task_ids).unsqueeze(-1)

        else:
            alphas = 1
            alpha_loss = 0

        with torch.no_grad():
            time6 = time.time()
            target_sample_info = self.pf.explore(next_obs,
                                                 neuron_masks=policy_device_masks,
                                                 return_log_probs=True)

            target_actions = target_sample_info["action"]
            target_log_probs = target_sample_info["log_prob"]

            cat_input = torch.cat([next_obs, target_actions], dim=1)

            target_q1_pred = self.target_qf1(cat_input, q1_device_masks)
            target_q2_pred = self.target_qf2(cat_input, q2_device_masks)

            min_target_q = torch.min(target_q1_pred, target_q2_pred)

            target_v_values = min_target_q - alphas[:, :] * target_log_probs
            time7 = time.time()
        """
        QF Loss
        """
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # There is no actual terminate in meta-world -> just filter all time_limit terminal
        # print("target_v_values,shape",target_v_values.shape)
        # print("rewards,shape",rewards[each_task_batch_idx].shape)
        
        q_target = rewards + self.discount * target_v_values
        #print("q_target",q_target.shape)
        qf1_loss = (reweight_coeff[:, :] *
                    ((q1_pred - q_target.detach()) ** 2)).mean()
        qf2_loss = (reweight_coeff[:, :] *
                    ((q2_pred - q_target.detach()) ** 2)).mean()

        assert q1_pred.shape == q_target.shape
        assert q2_pred.shape == q_target.shape
        time8 = time.time()
        cat_input = torch.cat([obs, new_actions], dim=1)
        q_new_actions = torch.min(
            self.qf1(cat_input,q1_device_masks),
            self.qf2(cat_input,q2_device_masks))

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
        time9 = time.time()
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
        time10 = time.time()
        

        # Information For Logger
        
        info = {}
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

        #all_info.append(info)

        return info

    def concat_task_masks(self,specific_mask_buffer,task_amount):
        batch_list =[]
        for layer_amount in range(len(specific_mask_buffer[0])):
            layer_list = [specific_mask_buffer[i][layer_amount].unsqueeze(0) for i in range(task_amount)]
            #layer_list[0].shape torch.Size([40])
            #print("layer_list[0].shape",layer_list[0].shape)
            layer_tensor = torch.cat(layer_list,dim=0).unsqueeze(1).to(self.device)
            batch_list.append(layer_tensor)
            #print("layer_tensor shape",layer_tensor.shape)layer_tensor shape torch.Size([400, 1])

        return batch_list

    def update_per_epoch(self, task_sample_index, task_scheduler, mask_buffer, epoch):
        time0 = time.time()
        mask_buffer_copy = copy.deepcopy(mask_buffer)
        #mask_buffer_copy = mask_buffer

        info = None
        dict2 = {}

        each_task_batch_size, update_idxes = self.get_batch_size_and_idx(task_scheduler)
        
        # policy_device_masks = self.concat_mask_tensors(task_scheduler.task_sample_num, 
        #                                               mask_buffer["Policy"], 
        #                                               each_task_batch_size, self.device)                                     

        # q1_device_masks = self.concat_mask_tensors(task_scheduler.task_sample_num, 
        #                                               mask_buffer["Q1"], 
        #                                               each_task_batch_size, self.device)

        # q2_device_masks = self.concat_mask_tensors(task_scheduler.task_sample_num, 
        #                                               mask_buffer["Q2"], 
        #                                               each_task_batch_size, self.device)

        policy_device_masks = self.concat_task_masks(mask_buffer_copy["Policy"],self.task_nums)
        time1 = time.time()
        q1_device_masks = self.concat_task_masks(mask_buffer_copy["Q1"],self.task_nums)
        time2 = time.time()
        q2_device_masks = self.concat_task_masks(mask_buffer_copy["Q2"],self.task_nums)
        timeafter = time.time()
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    self.task_nums,
                                                    task_sample_index=task_sample_index,
                                                    reshape=False)
            # Here, mask_buffer is all network types and all tasks.
            info = self.update(batch, task_sample_index, task_scheduler, 
                                         mask_buffer_copy,each_task_batch_size, update_idxes,
                                         policy_device_masks,q1_device_masks,q2_device_masks)


            
            self.logger.add_update_info(info)
        time_final = time.time()
        dict2["time0"] = time1 - time0
        dict2["time1"] = time2 - time1
        dict2["time2"] = timeafter - time2
        dict2["time_final"] = time_final - timeafter

        del policy_device_masks
        del q1_device_masks
        del q2_device_masks
        #del mask_buffer_copy
        wandb.log(info,step=epoch)
        wandb.log(dict2,step=epoch)


        # print(f'num_steps_can_sample: {self.replay_buffer.num_steps_can_sample()}')
