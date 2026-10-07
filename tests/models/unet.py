import torch
from torch import nn

from .unet_base import BaseRegressionModel

class UNetClassic(BaseRegressionModel):
    def __init__(self):
        super().__init__()

        b = 32
        self.first = nn.Sequential(
            nn.Conv2d(3, b, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
            nn.ReLU()
        )
        self.contractingC1 = nn.Sequential(
            nn.Conv2d(b, b, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU()
        )
        self.contractingC2 = self.contractingBlock(b)
        self.contractingC3 = self.contractingBlock(b * 2)

        self.contracting_neck = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2), # 16x32
            nn.Conv2d(b*4, b*8, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(b*8, b*8, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5)
        )
        # Scalars feature map injection
        self.expanding_neck = nn.Sequential(
            nn.ConvTranspose2d(b*8+2, b*4, kernel_size=2, stride=2),
            nn.ReLU()
        )

        # Skip connection x2
        self.expandingE3 = self.expandingBlock(b * 8)
        # Skip connection x2
        self.expandingE2 = self.expandingBlock(b * 4)
        # Skip connection x2
        self.expandingE1 = nn.Sequential(
            nn.Conv2d(b * 2, b, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Conv2d(b, 3, kernel_size=1, stride=1, padding=0),
        )

        # Module dicts for structuring different parts
        self.head = nn.ModuleDict({
            'first': self.first,
            'final': self.final
        })
        self.embed = nn.Identity()
        self.body = nn.ModuleDict({
            'contractingC1': self.contractingC1,
            'contractingC2': self.contractingC2,
            'contractingC3': self.contractingC3,
            'contracting_neck': self.contracting_neck,
            'expanding_neck': self.expanding_neck,
            'expandingE3': self.expandingE3,
            'expandingE2': self.expandingE2,
            'expandingE1': self.expandingE1
        })

    @staticmethod
    def contractingBlock(feature_channels):#X
        return nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2), # 16x32
            nn.Conv2d(feature_channels, feature_channels*2, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), # 32x64
            nn.ReLU(),
            nn.Conv2d(feature_channels*2, feature_channels*2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU()
        )

    @staticmethod
    def expandingBlock(feature_channels):
        return nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 2, feature_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_channels // 2, feature_channels // 4, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, X):
        x0D, x2D = X

        first = self.first(x2D)
        C1 = self.contractingC1(first)
        C2 = self.contractingC2(C1)
        C3 = self.contractingC3(C2)
        
        neck = self.contracting_neck(C3)
        scalar_map = x0D[:, :, None, None].expand(-1, -1, neck.shape[2], neck.shape[3])
        neck = torch.cat([neck, scalar_map], dim=1)
        neck = self.expanding_neck(neck)

        E3 = self.expandingE3(torch.cat([C3, neck], dim=1))
        E2 = self.expandingE2(torch.cat([C2, E3], dim=1))
        E1 = self.expandingE1(torch.cat([C1, E2], dim=1))
        final = self.final(E1)
        y = final.view(-1, x2D.shape[1], x2D.shape[2], x2D.shape[3])

        return y
