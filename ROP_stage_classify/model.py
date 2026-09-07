import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

        raise ValueError(f"Unsupported data_format: {self.data_format}")


class EnhancedMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = dim * mlp_ratio

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.pos = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim
        )
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(self.norm(x)))
        x = x + self.act(self.pos(x))
        return self.fc2(x)


class SpatialAttention(nn.Module):
    """Mask-aware spatial attention."""

    def __init__(self, dim, kernel_size=7, mask_strength=2.0):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
            ),
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # beta = softplus(mask_strength) > 0
        self.mask_strength = nn.Parameter(torch.tensor(float(mask_strength)))

    def forward(self, x, mask=None):
        x = self.norm(x)
        att = self.att(x)

        if mask is not None:
            mask = F.interpolate(
                mask.float(),
                size=att.shape[-2:],
                mode="area",
            ).clamp(0, 1)

            beta = F.softplus(self.mask_strength)
            att = att - beta * mask

        gate = torch.sigmoid(att)
        return self.proj(gate * self.v(x))


class CustomBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.1,
        mask_strength=2.0,
    ):
        super().__init__()

        self.attn = SpatialAttention(dim, kernel_size, mask_strength)
        self.mlp = EnhancedMLP(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        layer_scale = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x, mask=None):
        x = x + self.drop_path(
            self.layer_scale_1[:, None, None] * self.attn(x, mask)
        )
        x = x + self.drop_path(
            self.layer_scale_2[:, None, None] * self.mlp(x)
        )
        return x


class MaskedNormalizedAvgPool2d(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mask=None):
        if mask is None:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

        mask = F.interpolate(
            mask.float(),
            size=x.shape[-2:],
            mode="area",
        ).clamp(0, 1)

        weight = 1.0 - mask

        return (
            (x * weight).sum(dim=(2, 3))
            / weight.sum(dim=(2, 3)).clamp_min(self.eps)
        )


class Multi_ROP(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrain=True,
        drop_path=0.1,
        mask_strength=2.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.finetune = pretrain

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrain else None
        base_model = convnext_tiny(
            weights=weights,
            stochastic_depth_prob=drop_path,
        )

        self.backbone = nn.Sequential(*list(base_model.features)[:8])

        self.extra = CustomBlock(
            dim=768,
            kernel_size=11,
            drop_path=drop_path,
            mask_strength=mask_strength,
        )

        self.pool = MaskedNormalizedAvgPool2d()
        self.head = nn.Linear(768, num_classes)

        self.extra.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, mask=None):
        feat = self.backbone(x)
        feat = self.extra(feat, mask)
        feat = self.pool(feat, mask)
        return self.head(feat)




class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class Norm(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def layernorm(x, eps=1e-6):
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    pt = None
    for model_key in ['model', 'model_ema']:
        if model_key in checkpoint:
            pt = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if pt == None:
        pt = checkpoint
    model_static = model.state_dict()
    pt_ = {}

    for k, v in pt.items():
        if k in model_static:
            if 'head' not in k:
                pt_[k] = v
        else:
            print(f'{k} not in model')
    model_static.update(pt_)
    model.load_state_dict(model_static)

    print("Loading pretrained model for Generator from " + model_path)
    return model


if __name__ == '__main__':

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    cut = 8
    backbone = nn.Sequential(*list(model.features)[:cut])
#test
    x = torch.randn(1, 3, 256, 256)
    out = backbone(x)

    print(f"前{cut}层输出shape: {out.shape}")
    print(f"通道数: {out.shape[1]}")



