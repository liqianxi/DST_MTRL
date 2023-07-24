import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.networks.init as init


class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def apply_mask(self, task_masks):
        weight_mask = task_masks[0]
        bias_mask = task_masks[1]
        assert len(weight_mask) == len(self.base) + len(self.last),"incorrect weight mask number"
        assert len(bias_mask) == len(self.base),"incorrect bias mask number"

        # Apply element-wise product to each weight and bias of the base neural net.
        for layer_idx in len(self.base):
            self.base[layer_idx].weight = torch.mul(self.base[layer_idx].weight, weight_mask[layer_idx])
            self.base[layer_idx].bias = torch.mul(self.base[layer_idx].bias, bias_mask[layer_idx])
            
        self.last.weight = torch.mul(self.last.weight,weight_mask[-1])
        


    def forward(self, x, task_masks):

        out = self.base(x)


        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out

class MaskedNet(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        # print("self.base",self.base)
        # print("self.base fcs",self.base.fcs)
        """
        print("self.base",self.base)
        self.base MLPBase(
            (fc0): Linear(in_features=29, out_features=400, bias=True)
            (fc1): Linear(in_features=400, out_features=400, bias=True)
            )
        
        """
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        #self.append_fcs = []
        # for i, next_shape in enumerate(append_hidden_shapes):
        #     fc = nn.Linear(append_input_shape, next_shape)
        #     append_hidden_init_func(fc)
        #     self.append_fcs.append(fc)
        #     # set attr for pytorch to track parameters( device )
        #     self.__setattr__("append_fc{}".format(i), fc)
        #     append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x, neuron_masks):
        # Has examined, this forward should be correct - 0723 qianxi.
        mask_out = x
        for idx, layer in enumerate(self.base.fcs):
            mask_out = self.activation_func(layer(mask_out) * neuron_masks[idx])

        out = self.last(mask_out)
        return out


class FlattenNet(MaskedNet):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)


def null_activation(x):
    return x
 
class BootstrappedNet(Net):
    def __init__(self, output_shape, 
                 head_num = 10,
                 **kwargs ):
        self.head_num = head_num
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        super().__init__(output_shape = output_shape, **kwargs)

    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        out = super().forward(x)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        
        out = out.reshape(out_shape)

        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)

        out = out.gather(-1, idx).squeeze(-1)
        return out


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, idx ):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, idx)

class MaskGeneratorNet(nn.Module):
    def __init__(self,
            base_type, em_input_shape,
            em_hidden_shapes,
            hidden_shapes,
            num_layers,
            trajectory_encoder,
            pruning_ratio,
            device,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu):

        super().__init__()

        #Note: Embedding base is the network part that converts a full trajectory into
        # a D-dim vector.


        self.base = trajectory_encoder
        self.pruning_ratio = pruning_ratio
        self.device = device
        #Note: Embedding base is the network part that converts task onehot into
        # a D-dim vector.
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes).float().to(device)

        self.activation_func = activation_func

        self.num_layers = num_layers
        self.layer_neurons = hidden_shapes

        # assert self.em_base.output_shape == self.base.output_shape, \
        #     "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape # gating_input_shape: D in paper

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape, self.layer_neurons[0]).to(self.device) # D X neurons
        last_init_func( self.gating_weight_fc_0)
        
        for layer_idx in range(num_layers-2):
            # W_up (layer_neurons x D)
            gating_weight_cond_fc = nn.Linear(self.layer_neurons[layer_idx+1],
                                              gating_input_shape).to(self.device)

            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)

            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)
            
            #W_down (D X layer_neurons)
            gating_weight_fc = nn.Linear(gating_input_shape, self.layer_neurons[layer_idx+1]).to(self.device)
            last_init_func(gating_weight_fc)

            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            
            self.gating_weight_fcs.append(gating_weight_fc)

        # W_up (layer_neurons x D)
        self.gating_weight_cond_last = nn.Linear(self.layer_neurons[-1],
                                                 gating_input_shape).to(self.device) 
        module_hidden_init_func(self.gating_weight_cond_last)

        #W_down (D X layer_neurons)
        self.gating_weight_last = nn.Linear(gating_input_shape, self.layer_neurons[-1]).to(self.device)
        last_init_func( self.gating_weight_last )

    def keep_topk(self, tensor, pruning_ratio,neurons):
        # Keep how many neurons at each layer.
        k = int(neurons * (1 - pruning_ratio))

        # Pick the highest k values. Set the rest to zero.
        values, indices = torch.topk(tensor, k)
        output = torch.zeros_like(tensor)
        output.scatter_(dim=0, index=indices, src=values)
        return output

    def get_learnable_params(self):
        dict = self.__dict__["_modules"]
        param_list = []

        for key, value in dict.items():
            if key != "base":
                param = [i for i in value.parameters()]
                param_list += param
            
        return param_list

    def bound_tensor(self, array):
        max_value = torch.max(array)
        min_value = torch.min(array)
        res_array = (array - min_value) / (max_value - min_value)

        return res_array



    def forward(self, x, embedding_input):
        # Here x is a trajectory of shape [traj_length, dim_of_each_state]
        # Return weights for visualization

        # Trajectory encoder embedding
        out = self.base.encode_lstm(x)

        # Task one hot embedding
        embedding = self.em_base(embedding_input)

        # Element wise multi
        embedding = embedding * out

        task_probs_masks = []

        activated = self.activation_func(embedding).to(self.device)


        #print("activated device",activated.device)
        # Next 3 lines output p^{l=1}
        # Attention:
        # Once we have the output feature, we first pick top k based on the pruning ratio
        # the for the rest non-zero values, convert them between 0 and 1.
        raw_weight = self.gating_weight_fc_0(activated)  
        layer_neurons = self.layer_neurons[0]
        raw_weight = raw_weight.view(layer_neurons)

       # print("raw_weight 1",raw_weight)

        # Logic:
        # 1. First, convert to probability list
        # 2. prob list is used to calculate loss later, due to differentiablility.
        # 3. at the end, keep top k and convert to binary mask.
        raw_weight = self.bound_tensor(raw_weight)



        #print("raw_weight 1 after softmax",raw_weight)
        task_probs_masks.append(raw_weight)

        idx = 1
        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):

            # Next 6 lines will recover the dimension of the features to D X 1
            cond = gating_weight_cond_fc(raw_weight)# W_up (neurons x D) * p^l

            cond = cond * embedding # (W_up * p^l) * embedding
            cond = self.activation_func(cond) #RELU (cond)

            # Next, p^{l+1} = W_d^l(cond), generate raw weights.
            raw_weight = gating_weight_fc(cond) # W_down
            layer_neurons = self.layer_neurons[idx]
            raw_weight = raw_weight.view(layer_neurons)
            idx +=1
            #print("raw_weight",raw_weight)
            raw_weight = self.bound_tensor(raw_weight)
            #print("raw_weight after softmax",raw_weight)
            task_probs_masks.append(raw_weight)

        cond = self.gating_weight_cond_last(raw_weight)  # W_up (neurons x D) * p^l

        # print("cond",cond)
        # print("embedding 2",embedding)
        cond = cond * embedding # (W_up * p^l) * embedding

        #print("mul",cond)
        cond = self.activation_func(cond)  #RELU (cond)

        # W_down, generate the neuron mask for the last layer.
        raw_last_weight = self.gating_weight_last(cond) 
        #print("raw_last_weight",raw_last_weight)
        # Change the prob to [0,1].
        raw_last_weight = self.bound_tensor(raw_last_weight)
        #print("raw_last_weight after softmax",raw_last_weight)
        task_probs_masks.append(raw_last_weight)   

        task_binary_masks = []

        for each_task_probs_mask in task_probs_masks:
            pruned_mask = self.keep_topk(each_task_probs_mask, 
                                         self.pruning_ratio,
                                         len(each_task_probs_mask))

            task_binary_masks.append(torch.where(pruned_mask>0,
                                                 torch.ones(pruned_mask.shape),
                                                 torch.zeros(pruned_mask.shape)))

        # print("in gen:task_probs_masks",task_probs_masks)
        # print("in gen: task_binary_masks",task_binary_masks)
        return task_probs_masks, task_binary_masks