# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
from channel import Channel
from fis_modules import FIS_ImportanceAssessment, FIS_PowerMask, apply_power_mask


""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """


def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32,
                                    kernel_size=5, padding=2)  # padding size could be changed here
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            # k = torch.tensor(k)
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x, return_pre_norm: bool = False):
        # x = self.imgae_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            if return_pre_norm:
                return x
            x = self.norm(x)
            return x
        if return_pre_norm:
            raise ValueError('return_pre_norm=True is not supported when is_temp=True')
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        # x = self.imgae_normalization(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss




class DeepJSCC_FIS(nn.Module):
    """
    FIS-enhanced DeepJSCC with the SAME encoder/decoder as baseline (fair comparison).

    Role of FIS:
    - FIS computes an allocation map A(i,j) over the encoder latent feature map
      (mask/power control), based on:
        (i) latent-derived importance I(i,j),
        (ii) channel SNR (dB),
        (iii) optional rate_budget in [0,1].

    Usage patterns:
    - For training/evaluation with an external Channel module:
        encoded, decoded_clean, info = model(x, snr_db=..., rate_budget=..., return_info=True)
        encoded_noisy = channel(encoded)
        decoded = model.decoder(encoded_noisy)

    Notes:
    - FIS modules are fixed (not trained).
    - Encoder/decoder are trainable as in baseline.
    """

    def __init__(self, c, snr_db: float = 10.0, rate_budget: float = 1.0, P: float = 1.0):
        super().__init__()
        self.encoder = _Encoder(c=c, P=P)
        self.decoder = _Decoder(c=c)

        self.fis_importance = FIS_ImportanceAssessment()
        self.fis_alloc = FIS_PowerMask()

        # default conditions (can be overridden per forward call)
        self.default_snr_db = float(snr_db)
        self.default_rate_budget = float(rate_budget)

    def forward(self, x, snr_db: float = None, rate_budget: float = None, return_info: bool = False):
        if snr_db is None:
            snr_db = self.default_snr_db
        if rate_budget is None:
            rate_budget = self.default_rate_budget

        # Pre-normalization latent (so FIS sees content-dependent variations)
        z_pre = self.encoder(x, return_pre_norm=True)  # (B,2c,H',W')
        I, info_I = self.fis_importance(z_pre)         # (B,H',W')
        A, info_A = self.fis_alloc(I, snr_db=snr_db, rate_budget=rate_budget)

        # Apply mask/power and then normalize to meet average power constraint (same as baseline)
        z_mask = apply_power_mask(z_pre, A)
        z = self.encoder.norm(z_mask)

        # "clean" reconstruction (no channel) is sometimes useful for debugging
        x_hat_clean = self.decoder(z)

        if not return_info:
            return z, x_hat_clean

        info = {
            "snr_db": float(snr_db),
            "rate_budget": float(rate_budget),
            "I": I.detach(),
            "A": A.detach(),
            "I_stats": info_I,
            "A_stats": info_A,
        }
        return z, x_hat_clean, info


if __name__ == '__main__':
    model = DeepJSCC(c=20)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())
    print(y)
    print(model.encoder.norm)
    print(model.encoder.norm(y))
    print(model.encoder.norm(y).size())
    print(model.encoder.norm(y).size()[1:])