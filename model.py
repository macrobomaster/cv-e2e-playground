from tinygrad.nn import Conv2d, Linear
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
        self.l_out = Linear(512, 3, bias=False)

    def __call__(self, x: Tensor):
        x = self.c1(x)
        x = self.c_out(x).gelu()
        x = x.mean((2, 3))
        return self.l_out(x)
