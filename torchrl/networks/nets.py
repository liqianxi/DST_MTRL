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
                #print("x.shape",x.shape)x.shape torch.Size([1280, 19])
                # for idx, layer in enumerate(self.base.fcs):
                #     layer_weight = self.base.fcs[idx].weight # 400,19
                #     layer_bias = self.base.fcs[idx].bias

                #     mask_out = self.activation_func(torch.matmul(mask_out, layer_weight.t()*neuron_masks[2*idx]) + layer_bias*neuron_masks[2*idx+1])

                # out = torch.matmul(mask_out,self.last.weight.t() * neuron_masks[-2]+self.last.bias * neuron_masks[-1])
                # batch way

                """
                neuron_masks[2*idx] torch.Size([10, 40, 19])
                multi shape torch.Size([10, 40, 19])
                mask_out shape torch.Size([1280, 19])   
                """
                
                for idx, layer in enumerate(self.base.fcs):
                    mask_out = mask_out.reshape((10,128,mask_out.shape[-1]))
                    layer_weight = self.base.fcs[idx].weight.unsqueeze(0) #  layer_weight torch.Size([1, 40, 19])
                    layer_bias = self.base.fcs[idx].bias.unsqueeze(0) # layer_bias torch.Size([1, 40])

                    weight_apply_mask = (layer_weight*neuron_masks[2*idx]).permute(0,2,1)#torch.Size([10, 19, 40])
                    bias_apply_mask = (layer_bias*neuron_masks[2*idx+1])
                    bias_apply_mask_batched = bias_apply_mask.unsqueeze(1).repeat(1, 128, 1)
                    #print("bias_apply_mask",bias_apply_mask.shape) #torch.Size([10, 40])
                    #net tmp shape torch.Size([10, 1, 40, 19])
                    #print("multi shape",tmp.shape)
                    #print("mask_out shape",mask_out.shape) #[10,128,19]
                    tmp = torch.matmul(mask_out, weight_apply_mask)
                    #print("tmp",tmp.shape)tmp torch.Size([10, 128, 40])
                    output = self.activation_func(tmp + bias_apply_mask_batched)##:
                    #print("output",output.shape)
                    mask_out = output.reshape((10,128,output.shape[-1]))

                    # mask_out = mask_out.reshape((x.shape[0],mask_out.shape[-1]))

                last_weight_apply_mask = (self.last.weight * neuron_masks[-2]).permute(0,2,1)#torch.Size([10, 19, 40])
                last_bias_apply_mask = self.last.bias * neuron_masks[-1]
                last_bias_apply_mask_batched = last_bias_apply_mask.unsqueeze(1).repeat(1, 128, 1)
                # print("last_bias_apply_mask_batched",last_bias_apply_mask_batched.shape)
                # print("mask_out",mask_out.shape)

                out = torch.matmul(mask_out,last_weight_apply_mask) + last_bias_apply_mask_batched
                out = out.reshape(1280,out.shape[-1])
                # print("out",out.shape)

            else: 

                #self.fc1.weight = nn.Parameter(weights[:input_size * hidden_size].view(hidden_size, input_size))
                # self.base.fcs[0].weight = nn.Parameter(neuron_masks[0])
                # self.base.fcs[0].bias = nn.Parameter(neuron_masks[1])
                # self.base.fcs[1].weight = nn.Parameter(neuron_masks[2])
                # self.base.fcs[1].bias = nn.Parameter(neuron_masks[3])
                # self.base.fcs[2].weight = nn.Parameter(neuron_masks[4])
                # self.base.fcs[2].bias = nn.Parameter(neuron_masks[5])
                # self.last.weight = nn.Parameter(neuron_masks[6])
                # self.last.bias = nn.Parameter(neuron_masks[7])

                """
                # Manually perform linear operations using weights and biases
                    fc1_weight = self.fc1.weight
                    fc1_bias = self.fc1.bias
                    fc2_weight = self.fc2.weight
                    fc2_bias = self.fc2.bias

                    # Perform linear operations manually
                    hidden = torch.relu(torch.matmul(input, fc1_weight.t()) + fc1_bias)
                    output = torch.matmul(hidden, fc2_weight.t()) + fc2_bias
                """

                # for idx, layer in enumerate(self.base.fcs):
                #     mask_out = self.activation_func(layer(mask_out))

                for idx, layer in enumerate(self.base.fcs):
                    layer_weight = self.base.fcs[idx].weight # 400,19
                    layer_bias = self.base.fcs[idx].bias
                    #print("layer_weight",layer_weight.shape)
                    #print("layer_bias",layer_bias.shape)
                    #print("neuron_masks[2*idx]",neuron_masks[2*idx].shape)
                    new_weight = layer_weight*neuron_masks[2*idx]
                    #print("new_weight",new_weight.shape)
                    new_bias = layer_bias * neuron_masks[2*idx+1]
                    #print("new_bias",new_bias.shape)
                    #print("mask_out",mask_out.shape)
                    multi = torch.matmul(mask_out, new_weight.t())
                    #print("multi",multi.shape)

                    mask_out = self.activation_func( multi+ new_bias)

                #print("self.last.weight",self.last.weight.shape)
                #print("self.last.bias",self.last.bias.shape)
                #print("neuron_masks[-2]",neuron_masks[-2].shape)
                #print("neuron_masks[-1]",neuron_masks[-1].shape)
                tmp = self.last.weight* neuron_masks[-2]

                #print("tmp.shape",tmp.shape)
                out = torch.matmul(mask_out,tmp.t())+self.last.bias * neuron_masks[-1]
                #out = self.last(mask_out)

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
            use_trajectory_info,
            main_input_dim,
            main_out_dim,
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
        self.use_trajectory_info = use_trajectory_info

        # assert self.em_base.output_shape == self.base.output_shape, \
        #     "embedding should has the same dimension with base output for gated" 

        self.encoder = trajectory_encoder
        self.main_input_dim = main_input_dim
        self.main_out_dim = main_out_dim

        """
        assume 3 MLPs: 400,400,400

        input -> first layer:
            weight:(first_dim, input_dim)
            biases: first_dim

        first layer -> second layer:
            weight:(second_dim, first_dim)
            biases: second_dim
        
        second layer -> third layer:
            weight:(third_dim, second_dim)
            biases: third_dim    

        third layer -> output:
            weight:(third_dim, output_dim)
            biases: output_dim


        """
        result_all_neuron_amount = self.main_input_dim * self.layer_neurons[0]\
                                 + self.layer_neurons[0]\
                                 + self.layer_neurons[0] * self.layer_neurons[1]\
                                 + self.layer_neurons[1] + self.layer_neurons[2] * self.layer_neurons[1] \
                                 + self.layer_neurons[2]\
                                 + self.layer_neurons[2] * main_out_dim \
                                 + main_out_dim
        self.sigmoid = torch.nn.Sigmoid()


        if use_trajectory_info:

            self.generator_body = nn.Sequential(
                nn.Linear(self.encode_dimension+self.one_hot_result_dim, 256),  
                nn.ReLU(),  
                nn.Linear(256, result_all_neuron_amount)  
            ).to(device)
        else:
            self.generator_body = nn.Sequential(
                nn.Linear(10, 256),  
                nn.ReLU(),  
                nn.Linear(256, result_all_neuron_amount)  
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
       
        if self.use_trajectory_info:
            traj_encodings = self.encoder.encode_lstm(x)

            # Task one hot embedding
            embedding = self.mlp_layers(embedding_input).squeeze(1)
            #print(embedding.shape)torch.Size([4, 1, 32])
            # Element wise multi
            #print("embedding",embedding.shape)
            task_info_embedding = torch.cat([embedding, traj_encodings],dim=1)
        #print("task_info_embedding",task_info_embedding.shape)
        else: 
            task_info_embedding = embedding_input

        mask_vector = self.generator_body(task_info_embedding)

        task_binary_masks = []

        slice_index = 0
        # 1. input -> first layer weight:
        element_amount = self.main_input_dim*self.layer_neurons[0]
        weight = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(weight,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10, self.layer_neurons[0],self.main_input_dim)))
        slice_index += element_amount

        # 2. input -> first layer biases:
        element_amount = self.layer_neurons[0]
        biases = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(biases,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,element_amount)))
        slice_index += element_amount

        #3. first layer -> second layer weight:
        element_amount = self.layer_neurons[1]*self.layer_neurons[0]
        weight = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(weight,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,self.layer_neurons[1], self.layer_neurons[0])))
        slice_index += element_amount

        # 4. first layer -> second layer biases:
        element_amount = self.layer_neurons[1]
        biases = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(biases,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,element_amount)))
        slice_index += element_amount

        #5. second layer -> third layer weight:
        element_amount = self.layer_neurons[2]*self.layer_neurons[1]
        weight = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(weight,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,self.layer_neurons[2], self.layer_neurons[1])))
        slice_index += element_amount

        # 6. second layer -> third layer biases:
        element_amount = self.layer_neurons[2]
        biases = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        # print(biases)
        # print(biases.shape)
        # print(k)
        pruned_mask = self.gumbel_softmax(biases,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,element_amount)))
        slice_index += element_amount

        # 7. third layer -> output:
        element_amount = self.layer_neurons[2] * self.main_out_dim
        biases = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)
        pruned_mask = self.gumbel_softmax(biases,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,self.main_out_dim, self.layer_neurons[2])))
        slice_index += element_amount

        # 8. third layer -> output biases:
        element_amount = self.main_out_dim
        # print("element_amount",element_amount)
        # print("mask_vector",mask_vector.shape[1])
        # print("slice_index",slice_index)
        biases = mask_vector[:,slice_index:slice_index+element_amount]
        k = int(element_amount - element_amount * self.pruning_ratio)

        pruned_mask = self.gumbel_softmax(biases,k,hard=True)
        task_binary_masks.append(pruned_mask.reshape((10,element_amount)))
        slice_index += element_amount



        # # MLP layers
        # idx = 0
        # for layer_idx in range(len(self.layer_neurons)):
        #     neuron_amount = self.layer_neurons[layer_idx] * self.layer_neurons[layer_idx] # 400*400
        #     k = int(neuron_amount - neuron_amount * self.pruning_ratio)
            
        #     selected = mask_vector[:,idx:idx+neuron_amount]# all batch rows, selected columns(neurons).
        #     pruned_mask = self.gumbel_softmax(selected,k,hard=True)
        #     # pruned_mask = self.keep_topk(selected, 
        #     #                              self.pruning_ratio,
        #     #                              neuron_amount)
        #     task_binary_masks.append(pruned_mask.reshape((self.layer_neurons[layer_idx],self.layer_neurons[layer_idx])))

        #     idx += neuron_amount
        
        converted_list = []
        for task in range(len(task_binary_masks[0])):
            inner_list = []
            for i in range(len(task_binary_masks)):
                
                inner_list.append(task_binary_masks[i][task])
            converted_list.append(inner_list)

        #print([i.shape for i in task_binary_masks])
        #[torch.Size([10, 40, 19]), torch.Size([10, 40]), torch.Size([10, 40, 40]), torch.Size([10, 40]), torch.Size([10, 40, 40]), torch.Size([10, 40]), torch.Size([10, 8, 40]), torch.Size([10, 8])]
        
        return task_binary_masks,converted_list


