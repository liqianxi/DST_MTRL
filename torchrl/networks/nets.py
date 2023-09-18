import numpy as np
import torch, copy
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

        self.activation_func = activation_func
        append_input_shape = self.base.output_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x, neuron_masks,enable_mask=True):
        # Have examined, this forward should be correct - 0723 qianxi.
        #neuron_masks - (10,1,400)
        mask_out = x
        if enable_mask:
            if len(neuron_masks[0].shape) > 2:
                # batch way
                for idx, layer in enumerate(self.base.fcs):
                    output = self.activation_func(layer(mask_out))
                    output_reshape = output.reshape((10,128,output.shape[-1]))
                    mask_out = output_reshape * neuron_masks[idx]
                    mask_out = mask_out.reshape((x.shape[0],mask_out.shape[-1]))

            else: 
                for idx, layer in enumerate(self.base.fcs):
                    mask_out = self.activation_func(layer(mask_out)) * neuron_masks[idx]

            out = self.last(mask_out)

        else: 
            for idx, layer in enumerate(self.base.fcs):
                mask_out = self.activation_func(layer(mask_out))

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
            em_input_shape,
            hidden_shapes,
            num_layers,
            info_dim,
            pruning_ratio,
            trajectory_encoder,
            device):

        super().__init__()

        #Note: Embedding base is the network part that converts a full trajectory into
        # a D-dim vector.
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.encode_dimension = info_dim
        self.one_hot_result_dim = 64
        

        one_hot_mlp_hidden = 256
        # Define the MLP layers
        self.mlp_layers = nn.Sequential(
            nn.Linear(em_input_shape, one_hot_mlp_hidden),  # First MLP layer: input_size -> hidden_size
            nn.ReLU(),                           # Activation function
            nn.Linear(one_hot_mlp_hidden, self.one_hot_result_dim )  # Second MLP layer: hidden_size -> output_size
        ).to(device)

        self.num_layers = num_layers
        self.layer_neurons = hidden_shapes

        # assert self.em_base.output_shape == self.base.output_shape, \
        #     "embedding should has the same dimension with base output for gated" 

        self.encoder = trajectory_encoder

        result_all_neuron_amount = sum(self.layer_neurons)
        self.sigmoid = torch.nn.Sigmoid()
       
        self.generator_body = nn.Sequential(
            nn.Linear(self.encode_dimension+self.one_hot_result_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, 512),
            nn.ReLU(),                           
            nn.Linear(512, result_all_neuron_amount)  
        ).to(device)

        #print("self.generator_body",self.generator_body)

    def keep_topk(self, tensor, pruning_ratio,neurons):
        # Keep how many neurons at each layer.
        k = int(neurons - neurons * pruning_ratio)

        # Pick the highest k values. Set the rest to zero.
        values, indices = torch.topk(tensor, k)
        #print(values, indices)
        output = torch.zeros_like(tensor)
        ones = torch.ones_like(values)
        output.scatter_(dim=1, index=indices, src=ones)
        #print("output",output)
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


    def gumbel_softmax(self, logits, k, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)


        if hard:
            # Straight through.
            #index = y_soft.max(dim, keepdim=True)[1]
            index = torch.topk(y_soft, k, dim=-1)[1]

            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret



    def forward(self, x, embedding_input):
        # Here x is a trajectory of shape [traj_length, dim_of_each_state]
        # Return weights for visualization

        # Trajectory encoder embedding
        
        #print("traj_input",traj_input)
       

        traj_encodings = self.encoder.encode_lstm(x)

        # Task one hot embedding
        embedding = self.mlp_layers(embedding_input).squeeze(1)
        #print(embedding.shape)torch.Size([4, 1, 32])
        # Element wise multi
        #print("embedding",embedding.shape)
        task_info_embedding = torch.cat([embedding, traj_encodings],dim=1)
        #print("task_info_embedding",task_info_embedding.shape)

        mask_vector = self.generator_body(task_info_embedding) #mask_vector torch.Size([4, 20])

        #print("mask_vector",mask_vector.shape)

        task_binary_masks = []

        idx = 0
        for layer_idx in range(len(self.layer_neurons)):
            neuron_amount = self.layer_neurons[layer_idx]
            k = int(neuron_amount - neuron_amount * self.pruning_ratio)
            
            selected = mask_vector[:,idx:idx+neuron_amount]# all batch rows, selected columns(neurons).
            pruned_mask = self.gumbel_softmax(selected,k,hard=True).to("cpu")
            # pruned_mask = self.keep_topk(selected, 
            #                              self.pruning_ratio,
            #                              neuron_amount).to("cpu")
            task_binary_masks.append(pruned_mask)

            idx += neuron_amount
        
        converted_list = []
        for task in range(len(task_binary_masks[0])):
            inner_list = []
            for i in range(len(task_binary_masks)):
                
                inner_list.append(task_binary_masks[i][task])
            converted_list.append(inner_list)
        
        #print("converted_list",converted_list)
        return converted_list


