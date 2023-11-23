from itertools import chain

from tinygrad.nn import Conv2d, Linear, Embedding, BatchNorm2d
from tinygrad.tensor import Tensor

from shufflenet import ShuffleNetV2


class DWConvBlock:
    def __init__(self, dim):
        self.cv = Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.norm = BatchNorm2d(dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.norm(self.cv(x)).gelu()


class PWConvBlock:
    def __init__(self, dim, out_dim):
        self.cv = Conv2d(dim, out_dim, kernel_size=1, bias=False)
        self.norm = BatchNorm2d(out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.norm(self.cv(x)).gelu()


class ConvDecoder:
    def __init__(self, dim, out_dim):
        inner_dim = dim // 2

        self.cv1 = Conv2d(dim, dim, kernel_size=5, padding=2, bias=False)
        self.norm1 = BatchNorm2d(dim)
        self.cv2 = Conv2d(dim, inner_dim, kernel_size=5, padding=2, bias=False)
        self.norm2 = BatchNorm2d(inner_dim)
        self.cv_res = Conv2d(dim, inner_dim, kernel_size=1, bias=False)

        self.cv3 = DWConvBlock(inner_dim)
        self.cv_out = Linear(inner_dim, out_dim, bias=False)
        self.norm_out = BatchNorm2d(out_dim)

        self.l_out = Linear(out_dim * 484, out_dim)

    def __call__(self, x: Tensor):
        x = self.norm1(self.cv1(x)).pad2d((1, 1, 1, 1)).avg_pool2d(3, 1).gelu()
        res = self.cv_res(x)
        x = self.norm2(self.cv2(x)).gelu() + res

        x = self.cv3(x).permute(0, 2, 3, 1)
        x = self.norm_out(self.cv_out(x).permute(0, 3, 1, 2))
        return self.l_out(x.reshape(x.shape[0], -1))


class Head:
    def __init__(self, dim):
        self.color_emb = Embedding(2, dim)
        self.cv_joint = PWConvBlock(dim * 2, dim)

        # self.cv_det = ConvDecoder(dim, 1)
        # self.cv_reg = ConvDecoder(dim, 2)
        self.cv_out = ConvDecoder(dim, 3)

    def __call__(self, x: Tensor, color: Tensor):
        color = self.color_emb(color)[:, 0, :]
        color = color.reshape(*color.shape, 1, 1).expand(*color.shape, *x.shape[-2:])
        x = x.cat(color, dim=1)
        x = self.cv_joint(x)

        # det = self.cv_det(x)
        # reg = self.cv_reg(x)
        # return det.cat(reg, dim=1)

        return self.cv_out(x)


class Pooler:
    def __init__(self, dim, out_dim):
        self.cv_in = PWConvBlock(dim, out_dim)

        self.stage1 = [DWConvBlock(out_dim) for _ in range(1)]
        self.stage2 = [DWConvBlock(out_dim) for _ in range(2)]
        self.stage3 = [DWConvBlock(out_dim) for _ in range(3)]

        self.cv_out = Conv2d(out_dim * 3, out_dim, kernel_size=1, bias=False)
        self.norm_out = BatchNorm2d(out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.cv_in(x)

        y1 = x.sequential(self.stage1)
        y2 = x.sequential(self.stage2)
        y3 = x.sequential(self.stage3)
        y = y1.cat(y2, y3, dim=1)

        return (x + self.norm_out(self.cv_out(y))).gelu()


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
        self.backbone = ShuffleNetV2()
        self.pooler = Pooler(336, 96)
        self.head = Head(96)

    def __call__(self, img: Tensor, color: Tensor):
        x2, x3, x4 = self.backbone(img.permute(0, 3, 1, 2).float() / 255)

        x2 = x2.pad2d((1, 1, 1, 1)).avg_pool2d(3, 2)
        x4 = upsample(x4, 2)
        x = x2.cat(x3, x4, dim=1)

        x = self.pooler(x)

        return self.head(x, color)
