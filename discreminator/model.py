import torch
import torch.nn as nn

import dnnlib


class VGG_MLP(nn.Module):
    def __init__(self):
        super(VGG_MLP, self).__init__()
        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval()
        # VGG16をfreeze
        for param in vgg16.parameters():
            param.requires_grad = False
        self.backbone = vgg16
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        embedding = self.backbone(x)
        x = self.fc(embedding)
        return embedding, x
