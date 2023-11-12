from itertools import chain

from tinygrad.nn import Conv2d, Linear, Embedding, LayerNorm
from tinygrad.tensor import Tensor


class ConvBlock:
    def __init__(self, dim):
        self.dcv = Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=False,
        )
        self.n = LayerNorm(dim)
        self.pcv1 = Linear(dim, dim * 4)
        self.pcv2 = Linear(dim * 4, dim)

    def __call__(self, x: Tensor):
        res = x
        x = self.dcv(x).permute(0, 2, 3, 1)
        x = self.pcv1(self.n(x)).mish()
        x = self.pcv2(x).permute(0, 3, 1, 2)
        return x + res


class ConvEmbedding:
    def __init__(self, dim, out_dim):
        self.cvs = [ConvBlock(dim), ConvBlock(dim)]
        self.cv_out = Conv2d(dim, out_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm(out_dim)

    def __call__(self, x: Tensor):
        x = x.sequential(self.cvs)
        x = self.cv_out(x)
        return self.norm(x.mean((2, 3)))


class Head:
    def __init__(self, dim):
        self.cv = Conv2d(dim, 256, kernel_size=5)

        self.conv_emb = ConvEmbedding(dim, 512)
        self.color_emb = Embedding(2, 512)

        self.joint = Linear(512 * 2, 512)
        self.l_out = Linear(512, 4)

    def __call__(self, x: Tensor, color: Tensor):
        x = self.conv_emb(x)
        x = x.reshape(x.shape[0], -1)

        color = self.color_emb(color)[:, 0, :]
        x = self.joint(x.cat(color, dim=1)).mish()
        return self.l_out(x)


def upsample(x: Tensor, scale_factor: int):
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(
        *[1, 1, 1] + [scale_factor] * _lens
    )
    return (
        tmp.reshape(list(x.shape) + [scale_factor] * _lens)
        .permute(
            [0, 1]
            + list(chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)]))
        )
        .reshape([b, c] + [x * scale_factor for x in x.shape[2:]])
    )


class Model:
    def __init__(self):
        self.head = Head(256)

    def __call__(self, x: Tensor, color: Tensor):
        return self.head(x, color)
