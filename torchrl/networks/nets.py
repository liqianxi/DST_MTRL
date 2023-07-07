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

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)


def null_activation(x):
    return x

class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,

            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,

            # gated_hidden
            add_bn = True,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
             **kwargs ):

        super().__init__()

        self.base = base_type( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules
        print("self.num_layers",self.num_layers)
        assert 1==2
        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)
        
        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        
        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights = False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)
        #self.gating_fcs [Linear(in_features=400, out_features=256, bias=True), Linear(in_features=256, out_features=256, bias=True)]
        #print("self.gating_fcs",self.gating_fcs) 
        print("self.gating_fcs",self.gating_fcs)
        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]
        print("base_shape",base_shape)
        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))  # torch.Size([128, 10, 4])

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)  # torch.Size([128, 10, 2, 2])
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))
        # if case 4 layers:
        #[Linear(in_features=256, out_features=16, bias=True), Linear(in_features=256, out_features=16, bias=True)] 
        #[Linear(in_features=16, out_features=256, bias=True), Linear(in_features=32, out_features=256, bias=True)]
        # print(self.gating_weight_fcs, self.gating_weight_cond_fcs)
        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)# W_up (4*4 x D)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond) # W_down
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))
        
        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )  # torch.Size([128, 10, 2, 256])

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)

        if return_weights:
            return out, weights, last_weight
        return out


class FlattenModularGatedCascadeCondNet(ModularGatedCascadeCondNet):
    def forward(self, input, embedding_input, return_weights = False):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, embedding_input, return_weights = return_weights)

 
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
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,
            num_layers,
            module_hidden,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
             **kwargs ):

        super().__init__()

        #Note: Embedding base is the network part that converts a full trajectory into
        # a D-dim vector.
        #TODO: Shall be replaced by an encoder.
        self.base = base_type( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        
        #Note: Embedding base is the network part that converts task onehot into
        # a D-dim vector.
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func

        self.num_layers = num_layers
        self.layer_neurons = hidden_shapes

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape # gating_input_shape: D in paper

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape, layer_neurons) # D X neurons
        last_init_func( self.gating_weight_fc_0)
        
        for layer_idx in range(num_layers-2):
            # W_up (layer_neurons x D)
            gating_weight_cond_fc = nn.Linear(layer_neurons,
                                              gating_input_shape)

            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)

            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)
            
            #W_down (D X layer_neurons)
            gating_weight_fc = nn.Linear(gating_input_shape, layer_neurons)
            last_init_func(gating_weight_fc)

            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            
            self.gating_weight_fcs.append(gating_weight_fc)

        # W_up (layer_neurons x D)
        self.gating_weight_cond_last = nn.Linear(layer_neurons,
                                                 gating_input_shape) 
        module_hidden_init_func(self.gating_weight_cond_last)

        #W_down (D X layer_neurons)
        self.gating_weight_last = nn.Linear(gating_input_shape, layer_neurons)
        last_init_func( self.gating_weight_last )

    def forward(self, x, embedding_input, return_weights = False):
        # Return weights for visualization

        # Trajectory encoder embedding
        out = self.base(x)

        # Task one hot embedding
        embedding = self.em_base(embedding_input)

        # Element wise multi
        embedding = embedding * out

        weight_shape = embedding.shape[:-1] + torch.Size([layer_neurons])
        neuron_masks = []

        # Next 3 lines output p^{l=1}
        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))  
        raw_weight = raw_weight.view(weight_shape)
        softmax_weight = F.softmax(raw_weight, dim=-1)
        neuron_masks.append(softmax_weight)

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):

            # Next 6 lines will recover the dimension of the features to D X 1
            cond = torch.cat(softmax_weight, dim=-1)
            cond = gating_weight_cond_fc(cond)# W_up (neurons x D) * p^l
            cond = cond * embedding # (W_up * p^l) * embedding
            cond = self.activation_func(cond) #RELU (cond)

            # Next, p^{l+1} = W_d^l(cond), generate raw weights.
            raw_weight = gating_weight_fc(cond) # W_down
            raw_weight = raw_weight.view(weight_shape)

            # Here, shape the neuron_masks to [0,1]
            softmax_weight = F.softmax(raw_weight, dim=-1)
            neuron_masks.append(softmax_weight)
        
        cond = torch.cat(softmax_weight, dim=-1)
        cond = self.gating_weight_cond_last(cond)  # W_up (neurons x D) * p^l
        cond = cond * embedding # (W_up * p^l) * embedding
        cond = self.activation_func(cond)  #RELU (cond)

        # W_down, generate the neuron mask for the last layer.
        raw_last_weight = self.gating_weight_last(cond) 

        # Change the prob to [0,1].
        last_weight = F.softmax(raw_last_weight, dim = -1)
        neuron_masks.append(last_weight)

        single_neuron_mask_matrix = torch.cat(neuron_masks,0)

        return single_neuron_mask_matrix