# This one uses PointNav visual encoder as an encoder here and freeze it during traing.
# This one uses only first few layers of the visual encoder from PointGoal navigation to prevent information loss
# Only Decoder will be trained
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnetbn_3rd as resnet
# import resnetbn_3rd as resnet

def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        baseplanes: int = 32,
        ngroups: int = 32,
    ):
        super().__init__()

        self.backbone = resnet.se_resneXt50(
            3, baseplanes, ngroups
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.compression(x)
        return x




class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str):
        super().__init__()
        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        # print("BLOCK CONFIG", block_config)
        # print("CHANNEL CONFIG", channel_config)
        for _, (res, up_rate) in enumerate(block_config):
            # print("RES", res)
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                
                # print("CHANNEL in out", channel_config[res1], channel_config[res2])
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                # print("up_rate", up_rate)
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)
        self.last_conv = nn.Conv2d(channel_config[input_res], 3, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.block_mod(input)
        x = self.last_conv(x)
        return torch.sigmoid(x)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class AE(nn.Module):
    def __init__(
        self,
        input_res,
        dec_block_str,
        dec_channel_str,
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.input_res = input_res
        self.dec_block_str = dec_block_str
        self.dec_channel_str = dec_channel_str
        self.alpha = alpha
        self.lr = lr

        # Encoder architecture
        self.enc = ResNetEncoder()

        # Decoder Architecture
        self.dec = Decoder(self.input_res, self.dec_block_str, self.dec_channel_str)



    def forward(self, x):
        # For generating reconstructions during inference
        z = self.enc(x)
        decoder_out = self.dec(z)
        return decoder_out


if __name__ == "__main__":

    dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:512"


    ae = AE(256,
        dec_block_config_str,
        dec_channel_config_str,
    )
    checkpoint =  torch.load('/mnt/iusers01/fatpou01/compsci01/n70579mp/split-diffuse-vae/pretrained_models/se_resneXt50_pointnav.pt')
    # print("all keys", checkpoint.keys())
    ae.enc.load_state_dict(checkpoint, strict=False)
    print("SUCCESFULLY LOADED")
    sample = torch.randn(1, 3, 256, 256)
    out = ae.training_step(sample, 0)
    print(ae)
    print(out.shape)
