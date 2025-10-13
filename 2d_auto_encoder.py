import torch.nn as nn
from hypernetwork import Hypernet, TargetNet
import torch


class InPlaceABN(nn.Module):
    def __init__(self, out_channels, activation='leaky_relu'):
        super(InPlaceABN, self).__init__()
        self.bn = nn.InstanceNorm2d(out_channels)
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError()
    
    def forward(self, x):
        # print(x.shape)
        return self.act(self.bn(x))

class ConvBnReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU2D, self).__init__()
        self.conv  = nn.Conv2d(in_channels,out_channels,kernel_size,stride,pad,bias=False)
        self.bn = norm_act(out_channels)
    
    def forward(self,x):
        # import ipdb; ipdb.set_trace()
        return self.bn(self.conv(x))



class ConvBnReLU2DEncoder(nn.Module):
    def __init__(self,
                 CHANNELS,
                 norm_act=InPlaceABN):
        super().__init__()
        # CHANNELS = [10,20,40,80]

        self.conv5 = ConvBnReLU2D(CHANNELS[1], CHANNELS[2], stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU2D(CHANNELS[2], CHANNELS[2], norm_act=norm_act)

        self.conv7 = ConvBnReLU2D(CHANNELS[2], CHANNELS[3], stride=2, norm_act=norm_act)
        self.conv8 = ConvBnReLU2D(CHANNELS[3], CHANNELS[3], norm_act=norm_act)

    def forward(self,x): 
        conv6 = self.conv6(self.conv5(x))

        conv8 = self.conv8(self.conv7(conv6))
        # x = self.conv4(self.conv3(conv2))
        # return x, conv2
        return conv8, conv6

class ConvBnReLU2DEncoderTarget(TargetNet):
    def __init__(self,
                 CHANNELS,
                 norm_act=InPlaceABN):
        super().__init__()
        self.in_channels = CHANNELS[0]
        self.out_channels = CHANNELS[-1]
        self.func = ConvBnReLU2DEncoder(CHANNELS,norm_act)
    
    def forward(self,x):
        return self.func(x)
    
    def get_in_dims(self):
        return [self.in_channels]
    
    def get_out_dims(self):
        return [self.out_channels]

    def get_submodules(self):
        return [self.func]

class ConvBnReLU2DEncoderHyper(nn.Module):
    def __init__(self,ftask_dim,
                 CHANNELS,
                 norm_act=InPlaceABN,
                 weight_dim=128,
                 deriv_hidden_dim=32, 
                 driv_num_layers=2,
                 codec_hidden_dim=64, 
                 codec_num_layers=2,
                 num_layers=8):
        super().__init__()
        target = ConvBnReLU2DEncoderTarget(CHANNELS,norm_act)
        self.hyper = Hypernet(target, ftask_dim,weight_dim,deriv_hidden_dim,driv_num_layers,
                              codec_hidden_dim,codec_num_layers,num_layers)
        
    def forward(self,ftask,inputs,early_sup=False):
        return self.hyper(ftask,inputs,early_sup)



class MultiLayer2DEncoderShallow(nn.Module):
    def __init__(self, ftask_dim, in_channels=120, out_channels=120, norm_act=InPlaceABN):
        super(MultiLayer2DEncoderShallow,self).__init__()
        self.out_channels = out_channels
        CHANNELS = [10,20,40,80]

        # [64,64,60]
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(in_channels/2))

        # [128,128,30]
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels//2, in_channels//4, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(in_channels/4))

        # [128,128,10] conv 2
        self.conv2 = ConvBnReLU2D(in_channels//4, CHANNELS[0], kernel_size=1, pad=0, norm_act=norm_act)

        # [64,64,20] conv4
        self.conv3 = ConvBnReLU2D(CHANNELS[0], CHANNELS[1], stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU2D(CHANNELS[1], CHANNELS[1], norm_act=norm_act)
        
        # [32,32,40] conv6
        # [16,16,80] conv8
        self.encoder = ConvBnReLU2DEncoderHyper(ftask_dim,CHANNELS)

        
        # [32,32,40] conv9
        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS[3], CHANNELS[2], 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[2]))
        
        # [64,64,20] conv 10
        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS[2], CHANNELS[1], 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[1]))

        # [128,128,10]
        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS[1], CHANNELS[0], 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(CHANNELS[0]))
        
        # [128,128,30]
        self.conv12 = ConvBnReLU2D(CHANNELS[0], out_channels//4, kernel_size=1, pad=0, norm_act=norm_act)

        # [64,64,60]
        self.conv13 = ConvBnReLU2D(out_channels//4, out_channels//2, stride=2, norm_act=norm_act)
        self.conv14 = ConvBnReLU2D(out_channels//2, out_channels//2, norm_act=norm_act)

        # [32,32,120]
        self.conv15 = ConvBnReLU2D(out_channels//2, out_channels, stride=2, norm_act=norm_act)
        self.conv16 = ConvBnReLU2D(out_channels, out_channels, norm_act=norm_act)


    def forward(self, ftask,x):

        b,n,f,h,w = x.shape
        x = x.reshape(b*n,f,h,w)

        conv2 = self.conv2(self.conv1(self.conv0(x)))

        conv4 = self.conv4(self.conv3(conv2))

        conv8, conv6 = self.encoder(ftask,conv4.reshape(b,n,*conv4.shape[1:]))
        conv6 = conv6.reshape(b*n,*conv6.shape[2:])
        conv8 = conv8.reshape(b*n,*conv8.shape[2:])
        
        conv9 = self.conv9(conv8) + conv6
        del conv6

        conv10 = self.conv10(conv9) + conv4
        del conv4

        conv11 = self.conv11(conv10) + conv2
        del conv2

        out = self.conv16(self.conv15(self.conv14(self.conv13(self.conv12(conv11)))))
        return out.reshape(b,n,f,h,w)


if __name__ == "__main__":
    encoder = MultiLayer2DEncoderShallow(ftask_dim=2102,in_channels=120,out_channels=120)
    ftask = torch.rand(16,2102)
    x = torch.rand(16,1,120,64,64)
    out = encoder(ftask,x)
    print(out.shape)