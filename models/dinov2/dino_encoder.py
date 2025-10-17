import torch
import torch.nn as nn
import torch.nn.functional as F

class Dino_Encoder(nn.Module):
    def __init__(self, encoder='vitl'):
        super(Dino_Encoder, self).__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        self.dim = self.dino.blocks[0].attn.qkv.in_features
        print(f'Initialized DINOv2 {encoder} with feature dim {self.dim}')
        self.dino.eval()

    def forward(self, x):
        # x: (B, 3, T, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        features = self.dino.forward_features(x)  # (B*T, dim, h', w')
        _, D, h_feat, w_feat = features.shape
        features = features.reshape(B, T, D, h_feat, w_feat).permute(0, 2, 1, 3, 4)  # (B, dim, T, h', w')
        return features