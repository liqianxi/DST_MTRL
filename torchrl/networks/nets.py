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

        assert len(weight_mask) == len(self.base) + len(self.last),"incorrect weight mask number"


        # Apply element-wise product to each weight of the base neural net.
        for layer_idx in len(self.base):
            self.base[layer_idx].weight = torch.mul(self.base[layer_idx].weight, weight_mask[layer_idx])
            
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
            all_batch_size,
            task_amount,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):
        super().__init__()
        self.base = base_type(activation_func=activation_func, **kwargs)

        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.each_task_batch_size = int(all_batch_size / task_amount)
        self.task_amount = task_amount
        self.all_batch_size = all_batch_size

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x, neuron_masks,enable_mask=True):        
        mask_out = x
        if enable_mask:
            
            if len(neuron_masks[0].shape) > 2:                
                for idx, layer in enumerate(self.base.fcs):
                    mask_out = mask_out.view(self.task_amount, self.each_task_batch_size, -1)

                    layer_weight = self.base.fcs[idx].weight.unsqueeze(0) 
                    layer_bias = self.base.fcs[idx].bias.unsqueeze(0)

                    weight_apply_mask = (layer_weight*neuron_masks[idx]).permute(0,2,1)
                    bias_batched = layer_bias.unsqueeze(1).repeat(1, self.each_task_batch_size, 1)

                    output = self.activation_func(torch.bmm(mask_out, weight_apply_mask) + bias_batched)

                    mask_out = output.view(self.task_amount, self.each_task_batch_size, -1)

                last_weight_apply_mask = (self.last.weight * neuron_masks[-1]).permute(0,2,1)

                out = torch.bmm(mask_out,last_weight_apply_mask)
                out = out.view(self.all_batch_size, -1) + self.last.bias.repeat(self.all_batch_size, 1)

            else: 
                for idx, layer in enumerate(self.base.fcs):
                    new_weight = self.base.fcs[idx].weight * neuron_masks[idx]
                    mask_out = self.activation_func( torch.mm(mask_out, new_weight.t())+ self.base.fcs[idx].bias)

                tmp = self.last.weight* neuron_masks[-1]
                out = torch.mm(mask_out, tmp.t()) + self.last.bias

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
            info_dim,
            pruning_ratio,
            trajectory_encoder,
            use_trajectory_info,
            main_input_dim,
            main_out_dim,
            device,
            task_amount,
            one_hot_mlp_hidden,
            generator_mlp_hidden,
            one_hot_result_dim):

        super().__init__()
        self.to(device)
        #Note: Embedding base is the network part that 
        # converts a full trajectory into
        # a D-dim vector.
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.encode_dimension = info_dim
        self.one_hot_result_dim = one_hot_result_dim
        self.task_amount = task_amount

        self.layer_neurons = hidden_shapes
        self.use_trajectory_info = use_trajectory_info

        self.encoder = trajectory_encoder
        self.main_input_dim = main_input_dim
        self.main_out_dim = main_out_dim

        """
        assume 3 MLPs: 400,400,400

        input -> first layer:
            weight:(first_dim, input_dim)

        first layer -> second layer:
            weight:(second_dim, first_dim)
        
        second layer -> third layer:
            weight:(third_dim, second_dim)

        third layer -> output:
            weight:(output_dim,third_dim)


        """

        # The output dim of the last layer has to be the size of all weights and biases in the agent MLP.
        result_all_neuron_amount = self.sum_up_dim(self.layer_neurons, 
                                                   self.main_input_dim,
                                                   self.main_out_dim)

        if use_trajectory_info:
            # Traj onehot dim + trajectory encode dim.
            self.generator_body = nn.Sequential(
                nn.Linear(self.encode_dimension+em_input_shape, generator_mlp_hidden),  
                nn.ReLU(),  
                nn.Linear(generator_mlp_hidden, result_all_neuron_amount), 
            ).to(device)
        else:
            self.generator_body = nn.Sequential(
                nn.Linear(em_input_shape, generator_mlp_hidden),  
                nn.ReLU(),  
                nn.Linear(generator_mlp_hidden, result_all_neuron_amount),
            ).to(device)
    
    def sum_up_dim(self, layer_neurons, main_input, main_output):
        sum2 = 0
        tmp = [main_input] + layer_neurons + [main_output]
    
        for idx in range(len(tmp)-1):
            sum2 += tmp[idx] * tmp[idx+1]

        return sum2

    def keep_topk(self, tensor, pruning_ratio,neurons):
        # Keep how many neurons at each layer.
        k = int(neurons - neurons * pruning_ratio)

        # Pick the highest k values. Set the rest to zero.
        values, indices = torch.topk(tensor, k)
        output = torch.zeros_like(tensor,device=self.device)
        ones = torch.ones_like(values,device=self.device)
        output.scatter_(dim=1, index=indices, src=ones)

        return output

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

            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0).to(self.device)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def preserve_amount(self, element_amount):
        return int(element_amount - element_amount * self.pruning_ratio)

    def forward(self, x, embedding_input):


        """
        embedding_input torch.Size([1, 1])
        task_info_embedding torch.Size([1])
        
        """


        if self.use_trajectory_info:
            traj_encodings = self.encoder.encode_lstm(x)

            # Task one hot embedding
            embedding = embedding_input.squeeze(1)

            # Concat
            task_info_embedding = torch.cat([embedding, traj_encodings],dim=1)

        else: 
            task_info_embedding = embedding_input.squeeze(1)
        

        mask_vector = self.generator_body(task_info_embedding)

        if embedding_input.shape[1] == 1:
            mask_vector = mask_vector.unsqueeze(0)


        task_binary_masks = []

        # Initial values.
        slice_index = 0
        weight_out_dim = self.layer_neurons[0]
        weight_in_dim = self.main_input_dim

        weight_element_amount = weight_in_dim * weight_out_dim

        for layer_idx in range(len(self.layer_neurons)+1):
            # 0,1,2,3
            # Why +1 layers here?
            # Because you have 3 MLP layers, plus the weight of the output layer.
            # Generate weights masks:
            
            weight = mask_vector[:,slice_index:slice_index+weight_element_amount]
            # Use a differentiable way to keep top k elements as the mask.
            weight_pruned_mask = self.gumbel_softmax(weight, 
                        self.preserve_amount(weight_element_amount),
                        hard=True)
            
            slice_index += weight_element_amount

            # Append masked binary weight masks to the result list.
            task_binary_masks.append(weight_pruned_mask.reshape((self.task_amount, 
                                                                weight_out_dim,
                                                                weight_in_dim)))

            # Set dimension values for the next layer's weight.
            if layer_idx != len(self.layer_neurons):
                weight_in_dim = self.layer_neurons[layer_idx]

                if layer_idx != len(self.layer_neurons)-1:
                    weight_out_dim = self.layer_neurons[layer_idx+1]

                else:
                    weight_out_dim = self.main_out_dim
            
                weight_element_amount = weight_in_dim * weight_out_dim

        
        converted_list = []

        for task in range(len(task_binary_masks[0])):
            inner_list = []

            for i in range(len(task_binary_masks)):
                inner_list.append(task_binary_masks[i][task])

            converted_list.append(inner_list)
        
        return task_binary_masks, converted_list
        
        

