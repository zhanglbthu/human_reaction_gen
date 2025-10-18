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
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        
        x = x.view(B*T, C, W, H)  # (B*T, 3, W, H)
        
        features = self.dino.forward_features(x)
        
        cls_features = features["x_norm_clstoken"] # (B*T, dim)
        cls_features = cls_features.view(B, T, self.dim)  # (B, T, dim)
        return cls_features