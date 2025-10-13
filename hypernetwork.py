import os, sys
from typing import List

import einops
import torch
from torch import nn
import torch.nn.functional as F


class TargetNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    # in_dims and out_dims are not necessarily the same with actual model input/output dims.
    # but prev output dim must be the same with next input dim.
    def get_in_dims(self) -> List[int]:
        raise NotImplementedError

    def get_out_dims(self) -> List[int]:
        raise NotImplementedError

    def get_submodules(self) -> List[nn.Module]:
        raise NotImplementedError

    def get_submodule_names(self) -> List[str]:
        if not hasattr(self, 'submodule_names'):
            self.submodule_names = []
            for subm in self.get_submodules():
                for name, module in self.named_modules():
                    if subm is module:
                        self.submodule_names.append(name)
                        break
        return self.submodule_names

    def merge_submodule_weights(self, weight_dicts):
        '''
        convert the weight dict of the submodules to the weight dict of the target net.
        '''
        weight_dict = {}
        for sub_name, wd in zip(self.get_submodule_names(),weight_dicts):
            for k, v in wd.items():
                weight_dict[sub_name + '.' + k] = v

        return weight_dict

class MLP(nn.Module):
    '''
    A simple MLP with a variable number of layers and hidden dimensions.
    '''

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        if num_layers == 1:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
            self.fcs.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
            self.fcs.append(nn.Linear(hidden_dim, out_dim))
        self.activation = activation

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)  # Remove the activation function from the last layer
            else:
                x = self.activation(fc(x))
        return x


class ModuleEncoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.encoder = MLP(self.param_cnt, weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, weight_dict):
        weight_vec = torch.cat([weight_dict[k].reshape(-1, s.numel()) for k, s in self.name_shape_dict.items()], dim=-1)
        if weight_vec.ndim == 1:
            weight_vec = weight_vec[None]
        return self.encoder(weight_vec)


class ModuleDecoder(nn.Module):
    def __init__(self, target_net, weight_dim, hidden_dim, num_layers):
        super().__init__()
        self.name_shape_dict = {
            k: v.shape
            for k, v in target_net.named_parameters()
        }
        self.param_cnt = sum([v.numel() for v in self.name_shape_dict.values()])
        self.decoder = MLP(weight_dim, self.param_cnt, hidden_dim=hidden_dim, num_layers=num_layers)
        self.chunks = [v.numel() for v in self.name_shape_dict.values()]

    def forward(self, weight_vec):
        decoded_weights = self.decoder(weight_vec)
        weight_chunks = torch.split(
            decoded_weights, self.chunks, dim=-1
        )
        weight_dict = {
            k: chunk.reshape(-1, *v)
            for chunk, (k, v) in zip(weight_chunks, self.name_shape_dict.items())
        }
        return weight_dict


class OptSubBlock(nn.Module):
    '''
    a OptSubBlock is responsible for estimating the derivative of output w.r.t. the inputs, and the target net parameters.

    '''

    def __init__(self, ftask_dim, in_dim, out_dim, weight_dim,
                 hidden_dim, num_layers,
                 dl_din_way='direct',
                 dl_dw_way='slice',
                 **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.forward_net = MLP(in_dim + weight_dim, out_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.dout_din = MLP(in_dim + weight_dim, out_dim * in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.dout_dw = MLP(in_dim + weight_dim, out_dim * weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.dl_din_way = dl_din_way
        self.dl_dw_way = dl_dw_way

        if dl_din_way == 'slice':
            self.mm_mlp_in = MLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_din_way == 'full':
            self.mm_mlp_in = MLP(out_dim * (in_dim + 1), in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            assert dl_din_way == 'direct'
        
        if dl_dw_way == 'slice':
            self.mm_mlp_w = MLP(2 * out_dim, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        elif dl_dw_way == 'full':
            self.mm_mlp_w = MLP(out_dim * (weight_dim + 1), weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            assert dl_dw_way == 'direct'

    def pseudo_forward(self, z_in, weight_emb):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        out = self.forward_net(torch.cat([z_in, weight_emb], dim=-1))
        return out

    def pseudo_backward(self, z_in, weight_emb, dl_dout):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, '1 i -> n i', n=z_in.shape[0])
        dout_din = einops.rearrange(
            self.dout_din(torch.cat([z_in, weight_emb], dim=-1)),
            'n (o i) -> n o i', i=self.out_dim
        )
        dout_dw = einops.rearrange(
            self.dout_dw(torch.cat([z_in, weight_emb], dim=-1)),
            'n (o i) -> n o i', i=self.out_dim
        )

        if self.dl_din_way == 'direct':
            dl_din = einops.einsum(dl_dout, dout_din, 'n o, n i o -> n i')
        elif self.dl_din_way == 'slice':
            dl_din = self.mm_mlp_in(
                torch.cat([einops.repeat(dl_dout, 'n o -> n i o', i=dout_din.shape[1]), dout_din], dim=2),
            )[..., 0]
        elif self.dl_din_way == 'full':
            dl_din = self.mm_mlp_in(
                torch.cat([dl_dout, dout_din.flatten(1)], dim=1)
            )


        if self.dl_dw_way == 'direct':
            dl_dw = einops.einsum(dl_dout, dout_dw, 'n o, n i o -> n i')
        elif self.dl_dw_way == 'slice':
            dl_dw = self.mm_mlp_w(
                torch.cat([einops.repeat(dl_dout, 'n o -> n i o', i=dout_dw.shape[1]), dout_dw], dim=2)
            )[..., 0]
        elif self.dl_dw_way == 'full':
            dl_dw = self.mm_mlp_w(
                torch.cat([dl_dout, dout_dw.flatten(1)], dim=1)
            )

        return dl_din, dl_dw


class OptBlock(nn.Module):
    '''
    A OptBlock takes the input and output shape of each sub-block of the target network.
    it first does a peusdo forward mimicking the forward pass of the target network.
    stores the internal results, and mimicking the backward pass.

    target net must implement three functions returning the submodule, input shapes, and output shapes.

    '''

    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 *args, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.in_dims = target_net.get_in_dims()
        self.out_dims = target_net.get_out_dims()
        self.weight_dim = weight_dim

        self.opt_sub_blocks = nn.ModuleList([
            OptSubBlock(ftask_dim, in_dim, out_dim, weight_dim,
                        deriv_hidden_dim, driv_num_layers,
                        *args, **kwargs)
            for in_dim, out_dim in zip(self.in_dims, self.out_dims)
        ])

        self.forward_in = MLP(ftask_dim, self.in_dims[0], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)
        self.dloss_dout = MLP(ftask_dim, self.out_dims[-1], hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers)

    def forward(self, ftask, weight_dicts: List[dict], encoders: List[ModuleEncoder], decoders: List[ModuleDecoder]):
        z_ins = [self.forward_in(ftask)]
        dl_douts = [self.dloss_dout(ftask)]

        weight_embs = [
            encoder(weight_dict) for encoder, weight_dict in zip(encoders, weight_dicts)
        ]
        for weight_emb, opt_sub_block in zip(weight_embs, self.opt_sub_blocks):
            z_ins.append(opt_sub_block.pseudo_forward(z_ins[-1], weight_emb))

        dw_dicts = []
        for opt_sub_block, z_in, weight_emb, decoder in reversed(list(zip(self.opt_sub_blocks, z_ins, weight_embs, decoders))):
            dl_dout, dl_dw = opt_sub_block.pseudo_backward(z_in, weight_emb, dl_douts[-1])
            dl_douts.append(dl_dout)
            dw_dict = decoder(dl_dw)
            dw_dicts.append(dw_dict)

        dw_dicts = list(reversed(dw_dicts))

        return dw_dicts

class ParamLN(nn.Module):
    def __init__(self, target_net):
        super().__init__()
        self.ln = nn.ModuleList(
            [nn.LayerNorm(v.shape) for v in target_net.parameters()]
        )

    def forward(self, weight_dict):
        return {
            k: ln(v)
            for ln, (k, v) in zip(self.ln, weight_dict.items())
        }

class Hypernet(nn.Module):
    def __init__(self, target_net: TargetNet,
                 ftask_dim, weight_dim,
                 deriv_hidden_dim, driv_num_layers,
                 codec_hidden_dim, codec_num_layers,
                 num_layers, *args, **kwargs):
        super().__init__()
        self.target_net = target_net
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim

        self.encoders = nn.ModuleList([
            ModuleEncoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])
        self.decoders = nn.ModuleList([
            ModuleDecoder(target_module, weight_dim, codec_hidden_dim, codec_num_layers)
            for target_module in target_net.get_submodules()
        ])
        self.opt_blocks = nn.ModuleList([
            OptBlock(target_net, ftask_dim, weight_dim,
                     deriv_hidden_dim, driv_num_layers,
                     *args, **kwargs)
            for _ in range(num_layers)
        ])

        self.dynamic_lrs = nn.Parameter(torch.zeros(num_layers).fill_(-1e-2))
        self.layer_norms = nn.ModuleList(
            [ParamLN(submodule) for submodule in self.target_net.get_submodules()]
        )

    def forward_block(self, ftask, weight_dicts, opt_block):
        weight_upd_dicts = opt_block(ftask, weight_dicts, self.encoders, self.decoders)
        weight_upd_dicts = [
            ln(submodule)
            for ln, submodule in zip(self.layer_norms, weight_upd_dicts)
        ]
        return weight_upd_dicts

    def forward_blocks(self, ftask):
        weight_dicts = list(
            dict(submodule.named_parameters())
            for submodule in self.target_net.get_submodules()
        )
        final_weight_dicts = []
        for i, (opt_block, lr) in enumerate(zip(self.opt_blocks, self.dynamic_lrs)):
            weight_upd_dicts = self.forward_block(ftask, weight_dicts, opt_block)
            weight_dicts = [
                {k: v + lr * v_upd for (k, v), (_, v_upd) in zip(weight_dict.items(), weight_upd_dict.items())}
                for weight_dict, weight_upd_dict in zip(weight_dicts, weight_upd_dicts)
            ]

            weight_dict = self.target_net.merge_submodule_weights(weight_dicts)
            final_weight_dicts.append(weight_dict)
        return final_weight_dicts

  
    def forward(self, ftask, inputs, early_sup=False):
        final_weight_dicts = self.forward_blocks(ftask)
        # import ipdb
        # ipdb.set_trace()
        if early_sup:
            return torch.stack([
                torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(self.target_net, generated_weight, inputs)
                for generated_weight in final_weight_dicts])
        else:
            return torch.vmap(torch.func.functional_call, in_dims=(None, 0, 0))(self.target_net, final_weight_dicts[-1], inputs)


class Toy(TargetNet):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # note this is not necessarily the same as the output of the target network
    # it can be [4, 10] or [64, 64] whatever
    def get_in_dims(self):
        return [4, 10]

    # the previous module's out dim must be the same as next module's input dim.
    def get_out_dims(self):
        return [10, 5]

    def get_submodules(self):
        return [self.fc1, self.fc2]


if __name__ == '__main__':
    toy = Toy()
    hypernet = Hypernet(toy, 256, 128, 32, 2, 64, 2, 8)
    ftask = torch.rand(16, 256)
    inputs = torch.rand(16, 4)
    output = hypernet(ftask, inputs)
    outputs = hypernet(ftask, inputs, early_sup=True)

    print(output.shape, outputs.shape)

