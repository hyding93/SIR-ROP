import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
class MultiAV4(nn.Module):
    """
    Lightweight pretraining model.

    Pretraining targets:
        10 -> [1, 0]
        11 -> [1, 1]
        00 -> [0, 0]

    Backbone:
        ImageNet-pretrained ConvNeXt-Tiny

    Head:
        GAP
        -> LayerNorm
        -> Linear(768, 256)
        -> GELU
        -> Dropout
        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        num_classes=2,
        pretrain=False,
        hidden_dim=256,
        dropout=0.2
    ):
        super(MultiAV4, self).__init__()
        base_model = convnext_tiny
        cut = 8
        if pretrain:
            print("=================train from imagenet================")

            backbone = base_model(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            print("===============train from scratch================")
            backbone = base_model(
                weights=None
            )

        layers = list(backbone.features)[:cut]
        self.sn_unet = nn.Sequential(*layers)
        C_backbone = 768
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(C_backbone),
            nn.Linear(
                C_backbone,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                self.num_classes
            )
        )
    def forward(self, x):
        x = self.sn_unet(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    m= MultiAV4()
    x = torch.randn((2,3,224,224))
    out = m(x)
    print(m(x).shape)
