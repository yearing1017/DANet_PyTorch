import torch.nn as nn

#from drn import drn_d_22, drn_d_38
from .attention import ChannelAttentionModule, PositionAttentionModule


class DANet(nn.Module):
    def __init__(self, n_classes, inter_channel=512):
        super().__init__()

        # convolution before attention modules
        self.conv2pam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )
        self.conv2cam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        # attention modules
        self.pam = PositionAttentionModule(in_channels=inter_channel)
        self.cam = ChannelAttentionModule()

        # convolution after attention modules
        self.pam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())
        self.cam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())

        # output layers for each attention module and sum features.
        self.conv_pam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )
        self.conv_cam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, n_classes, 1)
        )


    def forward(self, x):

        # outputs from attention modules
        pam_out = self.conv2pam(x)
        pam_out = self.pam(pam_out)
        pam_out = self.pam2conv(pam_out)

        cam_out = self.conv2cam(x)
        cam_out = self.cam(cam_out)
        cam_out = self.cam2conv(cam_out)

        # segmentation result
        outputs = []
        feats_sum = pam_out + cam_out
        outputs = self.conv_out(feats_sum)
        #outputs.append(self.conv_pam_out(pam_out))
        #outputs.append(self.conv_cam_out(cam_out))

        return outputs
