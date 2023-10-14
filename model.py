from tinygrad.nn import Conv2d, Linear, Embedding
from tinygrad.tensor import Tensor


class ConvBlock:
    def __init__(self, c_in, c_out):
        self.c1 = Conv2d(c_in, c_out // 2, kernel_size=3, padding=1, bias=False)
        self.c_res = Conv2d(c_out // 2, c_out, kernel_size=1, bias=False)
        self.c2 = Conv2d(c_out // 2, c_out, kernel_size=3, padding=1, bias=False)

    def __call__(self, x: Tensor):
        x = self.c1(x).max_pool2d().gelu()
        residual = self.c_res(x)
        x = self.c2(x).gelu()
        return x + residual


class ConvEmbedding:
    def __init__(self, c_in, out_dim, reduction="max"):
        self.pre_conv = ConvBlock(c_in, out_dim // 2)
        self.post_conv = Conv2d(out_dim // 2, out_dim, kernel_size=3)
        assert reduction in ["max", "mean", "sum"], "reduction must be max, mean, or sum"
        self.reduction = reduction

    def __call__(self, x: Tensor):
        x = self.pre_conv(x)
        x = self.post_conv(x)
        if self.reduction == "max":
            x = x.max((2, 3))
        elif self.reduction == "mean":
            x = x.mean((2, 3))
        elif self.reduction == "sum":
            x = x.sum((2, 3))
        return x


class Head:
    def __init__(self):
        self.conv_emb = ConvEmbedding(256, 512)
        self.color_emb = Embedding(2, 512)
        self.joint = Linear(512 * 2, 512)
        self.l_out = Linear(512, 4, bias=False)

    def __call__(self, x: Tensor, color: Tensor):
        x = self.conv_emb(x)
        color = self.color_emb(color)[:, 0, :]
        x = self.joint(x.cat(color, dim=1)).leakyrelu()
        return self.l_out(x)
