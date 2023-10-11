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


class Head:
    def __init__(self):
        self.c1 = ConvBlock(512, 256)
        self.c_out = Conv2d(256, 512, kernel_size=3, bias=False)
        self.color = Embedding(2, 512)
        self.joint = Linear(512 * 2, 512)
        self.l_out = Linear(512, 4, bias=False)

    def __call__(self, x: Tensor, color: Tensor):
        x = self.c1(x)
        x = self.c_out(x)
        x = x.max((2, 3))
        color = self.color(color).reshape(x.shape[0], x.shape[1])
        x = self.joint(x.cat(color, dim=1)).leakyrelu()
        return self.l_out(x)
