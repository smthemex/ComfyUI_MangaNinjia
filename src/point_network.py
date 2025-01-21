import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from diffusers.models.modeling_utils import ModelMixin
class PointNet(ModelMixin):
    def __init__(
        self,
        conditioning_channels: int = 1,
        out_channels: Tuple[int] = (320, 640, 1280, 1280),
        downsamples: Tuple[int] = (6, 2, 2, 2)
    ):
        super(PointNet, self).__init__()
        
        self.blocks = nn.ModuleList()
        current_channels = conditioning_channels
        
        # 构造卷积块
        for out_channel, downsample in zip(out_channels, downsamples):
            layers = []
            for _ in range(downsample // 2):
                layers.append(nn.Conv2d(in_channels=current_channels, out_channels=out_channel, kernel_size=3, stride=2, padding=1))
                layers.append(nn.SiLU())
                current_channels = out_channel
            self.blocks.append(nn.Sequential(*layers))
    
    def forward(self, x):
        embeddings = []
        embedding = x
        for block in self.blocks:
            embedding = block(embedding)
            B, C, H, W = embedding.shape 
            embeddings.append(embedding.view(B, C, H * W).transpose(1, 2))
            # embeddings.append(embedding)
        return embeddings

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = PointNet().to(device)
    
    dummy_input = torch.randn(1, 1, 288, 512).to(device)  # Batch size = 1, Channels = 1, Height = 288, Width = 512
    embeddings = model(dummy_input)
    for i, embedding in enumerate(embeddings):
        print(f"Output at layer {i + 1}:", embedding.shape)