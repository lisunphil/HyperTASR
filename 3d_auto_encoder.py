import torch.nn as nn
from hypernetwork import Hypernet, TargetNet
import torch


class InPlaceABN(nn.Module):
    def __init__(self, out_channels, activation='leaky_relu'):
        super(InPlaceABN, self).__init__()
        self.bn = nn.InstanceNorm3d(out_channels)
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError()
    
    def forward(self, x):
        return self.act(self.bn(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))


class Conv3dTarget(TargetNet):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=3,
                 stride=1,
                 padding=1,
                 norm_act=InPlaceABN
                 ):
        super(Conv3dTarget,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv3d(in_channels, out_channels,
                              kernel_sizes, stride=stride, padding=padding, bias=False)
        self.bn = norm_act(out_channels)
    
    def forward(self,x):
        x = self.bn(self.conv(x))
        return x

    def get_in_dims(self):
        return [self.in_channels]
    
    def get_out_dims(self):
        return [self.out_channels]

    def get_submodules(self):
        return [self.conv]

class Conv3dHyper(torch.nn.Module):
    def __init__(self,
                 ftask_dim, 
                 in_channels,
                 out_channels,
                 weight_dim=128,
                 deriv_hidden_dim=32, 
                 driv_num_layers=2,
                 codec_hidden_dim=64, 
                 codec_num_layers=2,
                 num_layers=8,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm_act=InPlaceABN
                 ):
        super(Conv3dHyper,self).__init__()
        conv3d = Conv3dTarget(in_channels,out_channels,kernel_size,stride,padding,norm_act)
        self.hyper = Hypernet(conv3d, ftask_dim,weight_dim,deriv_hidden_dim,driv_num_layers,
                              codec_hidden_dim,codec_num_layers,num_layers)
        
    def forward(self, ftask, inputs, early_sup=False):
        return self.hyper(ftask,inputs,early_sup)


class HyperMultiLayer3DEncoderShallow(nn.Module):
    def __init__(self, condition_dim, in_channels=10, out_channels=64, norm_act=InPlaceABN, useHyper=[True,True,True,True,False,False,False]):
        super(HyperMultiLayer3DEncoderShallow, self).__init__()
        self.out_channels = out_channels
        self.useHyper = useHyper
        CHANNELS = [8, 16, 32, 64] # 0.58M
        # CHANNELS = [64, 128, 256, 512] # 18.60M
        # CHANNELS = [32, 64, 128, 256] # 4.65M (32+64+128=224)
        if useHyper[0]:
            self.conv0 = Conv3dHyper(condition_dim, in_channels, CHANNELS[0], kernel_size=3 , norm_act=norm_act)
        else:
            self.conv0 = ConvBnReLU3D(in_channels, CHANNELS[0], norm_act=norm_act)


        if useHyper[1]:
            self.conv1 = Conv3dHyper(condition_dim, CHANNELS[0], CHANNELS[1], kernel_size=3 ,stride=2, norm_act=norm_act)
        else:
            self.conv1 = ConvBnReLU3D(CHANNELS[0], CHANNELS[1], stride=2, norm_act=norm_act)
        if useHyper[2]:
            self.conv2 = Conv3dHyper(condition_dim, CHANNELS[1], CHANNELS[1], kernel_size=3 ,norm_act=norm_act)
        else:
            self.conv2 = ConvBnReLU3D(CHANNELS[1], CHANNELS[1], norm_act=norm_act)
            
        if useHyper[3]:
            self.conv3 = Conv3dHyper(condition_dim, CHANNELS[1], CHANNELS[2], kernel_size=3 ,stride=2, norm_act=norm_act)
        else:
            self.conv3 = ConvBnReLU3D(CHANNELS[1], CHANNELS[2], stride=2, norm_act=norm_act)
        if useHyper[4]:
            self.conv4 = Conv3dHyper(condition_dim, CHANNELS[2], CHANNELS[2], kernel_size=3 ,norm_act=norm_act)
        else:
            self.conv4 = ConvBnReLU3D(CHANNELS[2], CHANNELS[2], norm_act=norm_act)

        if useHyper[5]:
            self.conv5 = Conv3dHyper(condition_dim, CHANNELS[2], CHANNELS[3], kernel_size=3 ,stride=2, norm_act=norm_act)
        else:
            self.conv5 = ConvBnReLU3D(CHANNELS[2], CHANNELS[3], stride=2, norm_act=norm_act)
        if useHyper[6]:
            self.conv6 = Conv3dHyper(condition_dim, CHANNELS[3], CHANNELS[3], kernel_size=3 ,norm_act=norm_act)
        else:
            self.conv6 = ConvBnReLU3D(CHANNELS[3], CHANNELS[3], norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(CHANNELS[3], CHANNELS[2], 3, padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[2]))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(CHANNELS[2], CHANNELS[1], 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[1]))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(CHANNELS[1], CHANNELS[0], 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[0]))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv3d(CHANNELS[0], out_channels, 1, stride=1, padding=0, bias=True)
        

    def forward(self, condition_x, x):
        voxel_list = [] # to store multi-scale features
        voxel_list.append(x)
        # print(self.useHyper)
        # conv0 = self.conv0(x) 
        if self.useHyper[0]:
            conv0 = self.conv0(condition_x,x)
        else:
            conv0 = self.conv0(x) 

        if self.useHyper[1]:
            conv1 = self.conv1(condition_x,conv0)
        else:
            conv1 = self.conv1(conv0) 

        if self.useHyper[2]:
            conv2 = self.conv2(condition_x,conv1) # 16, 50^3
        else:
            conv2 = self.conv2(conv1) 
        del conv1

        if self.useHyper[3]:
            conv3 = self.conv3(condition_x,conv2)
        else:
            conv3 = self.conv3(conv2) 

        if self.useHyper[4]:
            conv4 = self.conv4(condition_x,conv3) # 32, 25^3
        else:
            conv4 = self.conv4(conv3) 
        del conv3


        if self.useHyper[5]:
            conv5 = self.conv5(condition_x,conv4)
        else:
            conv5 = self.conv5(conv4) 

        if self.useHyper[6]:
            x = self.conv6(condition_x,conv5) # 64, 13^3
        else:
            x = self.conv6(conv5) 
        del conv5

        x = conv4 + self.conv7(x) # 32, 25^3
        voxel_list.append(x)
        del conv4
        x = conv2 + self.conv9(x)
        voxel_list.append(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.conv_out(x)
        return x, voxel_list
    

if __name__ == "__main__":
    encoder = HyperMultiLayer3DEncoderShallow(condition_dim=2102,in_channels=10,out_channels=20)
    condition_x = torch.rand(1,2102)
    x = torch.rand(1,10,100,100,100)
    out, multi_scale_voxel_list = encoder(condition_x,x)
    print(out.shape)