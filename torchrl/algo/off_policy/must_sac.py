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
                 mask_update_itv,
                 batch_size,
                 temp_reweight=False,
                 grad_clip=True,
                 **kwargs):

        super().__init__(**kwargs)

        self.task_nums = task_nums
        self.mask_update_itv = mask_update_itv
        self.batch_size=batch_size

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

    def clip_by_window(self,list_of_trajs,window_length):
        if len(list_of_trajs) > window_length:
            return list_of_trajs[-window_length:]
        return list_of_trajs

    def get_masks(self, sampled_task_amount,all_task_amount):
        recent_window = self.recent_traj_window   
        for t_id in range(all_task_amount):
            self.state_trajectory[t_id] = self.clip_by_window(self.state_trajectory[t_id],recent_window)    

        all_dict = {}
        for each_net in ["Policy","Q1","Q2"]:      
            

            task_onehot_batch = torch.stack([self.one_hot_map[i].squeeze(0) for i in range(all_task_amount)]).to(self.device)

            generator = self.policy_mask_generator
            if each_net == "Q1":
                generator = self.qf1_mask_generator
            elif each_net == "Q2":
                generator = self.qf2_mask_generator
            if self.use_sl_loss:
                task_traj_batch = self.sample_update_data(self.device)
                _,batch_task_binary_masks = generator(task_traj_batch, task_onehot_batch)
            else: 
                task_traj_batch = None
                _,batch_task_binary_masks = generator(task_traj_batch, task_onehot_batch)

            tmp_dict = {}
            """
            single_msk.shape torch.Size([40, 19])
            single_msk.shape torch.Size([40])
            single_msk.shape torch.Size([40, 40])
            single_msk.shape torch.Size([40])
            single_msk.shape torch.Size([40, 40])
            single_msk.shape torch.Size([40])
            single_msk.shape torch.Size([8, 40])
            single_msk.shape torch.Size([8])
            """
            for task in range(sampled_task_amount):
                task_mask_list = []
                for each_layer in range(len(batch_task_binary_masks[0])):
                    single_msk = batch_task_binary_masks[task][each_layer]
                    task_mask_list.append(single_msk)

                tmp_dict[task] = task_mask_list


            all_dict[each_net] = tmp_dict

        return all_dict

    def get_batch_size_and_idx(self,task_scheduler):
        batch_size = self.batch_size
        each_task_batch_size = 0
        if task_scheduler.task_sample_num == 10:
            update_idxes = np.ones(10, dtype=np.int32) * (batch_size // 10)
            each_task_batch_size = batch_size // 10
        elif task_scheduler.task_sample_num == 50:
            update_idxes = np.ones(50, dtype=np.int32) * (batch_size // 50)
            each_task_batch_size = batch_size // 50
        else:
            # update_idxes = (task_scheduler.p * batch_size).astype(np.int32)
            # update_idxes[-1] = batch_size - np.sum(update_idxes[:-1])
            update_idxes = np.ones(self.task_nums, dtype=np.int32) * (batch_size // self.task_nums)
            each_task_batch_size = batch_size // self.task_nums

        return each_task_batch_size, update_idxes

    def update(self, batch, task_sample_index, task_scheduler, mask_buffer,
               each_task_batch_size, update_idxes,
               policy_device_masks,q1_device_masks,q2_device_masks, epoch, disable_gen_grad
               ):

        self.training_update_num += 1

        time0 = time.time()

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

        time1 = time.time()

        embedding_inputs = batch["embedding_inputs"]

        embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)
        embedding_inputs = torch.cat([embedding_inputs[:update_idxes[i], i, :] for i in range(task_scheduler.num_tasks)])

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
        sample_info = self.pf.explore(obs, 
                                      neuron_masks=policy_device_masks,
                                      return_log_probs=True)

        time2 = time.time()

        mean = sample_info["mean"] #1280*xx
        log_std = sample_info["log_std"]
        new_actions = sample_info["action"]
        log_probs = sample_info["log_prob"]

        cat_input = torch.cat([obs, actions], dim=1)

        q1_pred = self.qf1(cat_input, q1_device_masks)
        q2_pred = self.qf2(cat_input, q2_device_masks)

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
            alpha_loss.backward(retain_graph=True)
            self.alpha_optimizer.step()


            alphas = torch.gather(self.log_alpha.exp(), 0, task_ids).unsqueeze(-1)

        else:
            alphas = 1
            alpha_loss = 0

        time3 = time.time()

        with torch.no_grad():
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

        time4 = time.time()
        """
        QF Loss
        """
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # There is no actual terminate in meta-world -> just filter all time_limit terminal
        
        q_target = rewards + self.discount * target_v_values
        #print("q_target",q_target.shape)
        qf1_loss = (reweight_coeff[:, :] *
                    ((q1_pred - q_target.detach()) ** 2)).mean()
        qf2_loss = (reweight_coeff[:, :] *
                    ((q2_pred - q_target.detach()) ** 2)).mean()

        assert q1_pred.shape == q_target.shape
        assert q2_pred.shape == q_target.shape

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

        """
        Update Networks
        """

        # if disable_gen_grad:
        #     print("disable_gen_grad")
        # self.policy_mask_generator.generator_body.requires_grad = False
        # self.qf1_mask_generator.generator_body.requires_grad = False
        # self.qf2_mask_generator.generator_body.requires_grad = False
        # self.policy_mask_generator.encoder.requires_grad = False
        # self.qf1_mask_generator.encoder.requires_grad = False
        # self.qf2_mask_generator.encoder.requires_grad = False

        # else: 
        #     print("enable_gen_grad")
        # self.policy_mask_generator.generator_body.requires_grad = True
        # self.qf1_mask_generator.generator_body.requires_grad = True
        # self.qf2_mask_generator.generator_body.requires_grad = True
        # self.policy_mask_generator.encoder.requires_grad = True
        # self.qf1_mask_generator.encoder.requires_grad = True
        # self.qf2_mask_generator.encoder.requires_grad = True

        policy_optimizer = self.pf_optimizer
        q1_optimizer = self.qf1_optimizer
        q2_optimizer = self.qf1_optimizer
        if disable_gen_grad:
            policy_optimizer = self.pf_optimizer_with_gen
            q1_optimizer = self.qf1_optimizer_with_gen
            q2_optimizer = self.qf2_optimizer_with_gen


        policy_optimizer.zero_grad()

        time5 = time.time()

        policy_loss.backward(retain_graph=True)
        #print("grad",torch.sum(list(self.policy_mask_generator.generator_body)[0].weight.grad))

        if self.grad_clip:
            pf_norm = torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.policy_mask_generator.generator_body.parameters(),1)
            torch.nn.utils.clip_grad_norm_(self.policy_mask_generator.encoder.parameters(),1)
            #torch.nn.utils.clip_grad_norm_(self.policy_mask_generator.mlp_layers.parameters(),1)



        policy_optimizer.step()
        # Dont know what is this.
        # self.pf.apply(rezero_weights)

        q1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        if self.grad_clip:
            qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.qf1_mask_generator.generator_body.parameters(),1)
            torch.nn.utils.clip_grad_norm_(self.qf1_mask_generator.encoder.parameters(),1)
            #torch.nn.utils.clip_grad_norm_(self.qf1_mask_generator.mlp_layers.parameters(),1)

        q1_optimizer.step()
        # self.qf1.apply(rezero_weights)

        q2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
    
        if self.grad_clip:
            qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.qf2_mask_generator.generator_body.parameters(),1)
            torch.nn.utils.clip_grad_norm_(self.qf2_mask_generator.encoder.parameters(),1)
            #torch.nn.utils.clip_grad_norm_(self.qf2_mask_generator.mlp_layers.parameters(),1)

        q2_optimizer.step()
        # self.qf2.apply(rezero_weights)

        self._update_target_networks()
        # Information For Logger
        time6 = time.time()
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
        inner_dict = {"sac_diff0":time1-time0,
                      "sac_diff1":time2-time1,
                      "sac_diff2":time3-time2,
                      "sac_diff3":time4-time3,
                      "sac_diff4":time5-time4,
                      "sac_diff5":time6-time5}


        return info,inner_dict

    def concat_task_masks(self,specific_mask_buffer,task_amount,detach_mask=False):
        batch_list =[]

        for layer_amount in range(len(specific_mask_buffer[0])):
            layer_list = [specific_mask_buffer[i][layer_amount].unsqueeze(0) for i in range(task_amount)]

            layer_tensor = torch.cat(layer_list,dim=0).to(self.device)

            #layer_tensor = layer_tensor.detach()
            if detach_mask:
                layer_tensor = layer_tensor.detach()

            batch_list.append(layer_tensor)

        return batch_list

    def update_per_epoch(self, task_sample_index, task_scheduler, mask_buffer, epoch, use_trajectory_info):
        info = None
        time0 = time.time()
        each_task_batch_size, update_idxes = self.get_batch_size_and_idx(task_scheduler)
        mask_buffer_copy = copy.deepcopy(mask_buffer)
        time1 = time.time()
        print("time1",time1-time0)
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0
        inner_dict_sum = {"sac_diff0":0,
                      "sac_diff1":0,
                      "sac_diff2":0,
                      "sac_diff3":0,
                      "sac_diff4":0,
                      "sac_diff5":0}

        diff5_list = []
        if epoch % self.mask_update_itv != 0:
            # If not update epoch, get mask once and reuse
            mask_buffer_copy = self.get_masks(self.task_nums,self.task_nums)
            policy_device_masks = self.concat_task_masks(mask_buffer_copy["Policy"],self.task_nums,detach_mask = True)

            q1_device_masks = self.concat_task_masks(mask_buffer_copy["Q1"],self.task_nums,detach_mask = True)

            q2_device_masks = self.concat_task_masks(mask_buffer_copy["Q2"],self.task_nums,detach_mask = True)

        #
        for opti_time in range(self.opt_times):
            disable_gen_grad=True


            if epoch % self.mask_update_itv == 0:
                # If update epoch, get mask every optim time.
                mask_buffer_copy = self.get_masks(self.task_nums,self.task_nums)
                disable_gen_grad=False

                inside_ckpt1 = time.time()

                policy_device_masks = self.concat_task_masks(mask_buffer_copy["Policy"],self.task_nums,detach_mask = False)

                q1_device_masks = self.concat_task_masks(mask_buffer_copy["Q1"],self.task_nums,detach_mask = False)

                q2_device_masks = self.concat_task_masks(mask_buffer_copy["Q2"],self.task_nums,detach_mask = False)

                inside_ckpt2 = time.time()

                time3 += inside_ckpt2 - inside_ckpt1

            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    self.task_nums,
                                                    task_sample_index=task_sample_index,
                                                    reshape=False)

            inside_ckpt3 = time.time()                                      
            # Here, mask_buffer is all network types and all tasks.
            # December. This consumes 99% of the training time.
            info,inner_dict = self.update(batch, task_sample_index, task_scheduler, 
                                         mask_buffer_copy,each_task_batch_size, update_idxes,
                                         policy_device_masks,q1_device_masks,q2_device_masks, epoch,disable_gen_grad)

            for key in inner_dict.keys():
                inner_dict_sum[key] +=inner_dict[key]

            diff5_list.append(inner_dict["sac_diff5"])

            self.logger.add_update_info(info)
            inside_ckpt4 = time.time()    
            time5 += inside_ckpt4 - inside_ckpt3 

        time6 = time.time()
        # Avoid frequent access and write shared memory.
        if epoch % self.mask_update_itv == 0:
            for each_net in ["Policy","Q1","Q2"]:  
                local_mask_update_dict = {}
                for i in range(self.task_nums):
                    new_mask_list = [msk_tensor.detach().to("cpu") for msk_tensor in mask_buffer_copy[each_net][i]]
                    local_mask_update_dict[i] = new_mask_list
                mask_buffer[each_net].update(local_mask_update_dict)

        time7 = time.time()
        print("inner_dict_sum",inner_dict_sum)
        print("diff5_list",diff5_list)

        print("time3",time3)
        print("time5",time5)
        print("time7",time7-time6)
        gen_weight_change = torch.sum([param for param in self.policy_mask_generator.generator_body.parameters()][-1].data)
        print("gen_weight_change",gen_weight_change)
        info["gen_weight_change"] = gen_weight_change
        del mask_buffer_copy
        del each_task_batch_size, update_idxes
        del policy_device_masks, q1_device_masks,q2_device_masks

        wandb.log(info,step=epoch)

        
