from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from itertools import chain
from math import inf


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def get_variant_multiples(variant):
    return {
        "n": (0.33, 0.25, 2.0),
        "s": (0.33, 0.50, 2.0),
        "m": (0.67, 0.75, 1.5),
        "l": (1.0, 1.0, 1.0),
        "x": (1, 1.25, 1.0),
    }.get(variant, None)


class Upsample:
    def __init__(self, scale_factor: int, mode: str = "nearest") -> None:
        assert mode == "nearest"  # only mode supported for now
        self.mode = mode
        self.scale_factor = scale_factor

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape) > 2 and len(x.shape) <= 5
        (b, c), _lens = x.shape[:2], len(x.shape[2:])
        tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(
            *[1, 1, 1] + [self.scale_factor] * _lens
        )
        return (
            tmp.reshape(list(x.shape) + [self.scale_factor] * _lens)
            .permute(
                [0, 1]
                + list(
                    chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)])
                )
            )
            .reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])
        )


class Conv_Block:
    def __init__(
        self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None
    ):
        self.conv = Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding, dilation),
            bias=False,
            groups=groups,
            dilation=dilation,
        )
        self.bn = BatchNorm2d(c2, eps=0.001)

    def __call__(self, x):
        return self.bn(self.conv(x)).silu()


class Bottleneck:
    def __init__(
        self, c1, c2, shortcut: bool, g=1, kernels: list = (3, 3), channel_factor=0.5
    ):
        c_ = int(c2 * channel_factor)
        self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
        self.cv2 = Conv_Block(
            c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g
        )
        self.residual = c1 == c2 and shortcut

    def __call__(self, x):
        return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))


class C2f:
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        self.c = int(c2 * e)
        self.cv1 = Conv_Block(c1, 2 * self.c, 1)
        self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
        self.bottleneck = [
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                kernels=[(3, 3), (3, 3)],
                channel_factor=1.0,
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        y = self.cv1(x).chunk(2, 1)
        y.extend(m(y[-1]) for m in self.bottleneck)
        z = y[0]
        for i in y[1:]:
            z = z.cat(i, dim=1)
        return self.cv2(z)


class SPPF:
    def __init__(self, c1, c2, k=5):
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_Block(c1, c_, 1, 1, padding=None)
        self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=None)

        self.maxpool = lambda x: x.pad2d(
            (k // 2, k // 2, k // 2, k // 2), value=-inf
        ).max_pool2d(kernel_size=k, stride=1)

    def __call__(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        return self.cv2(x.cat(x2, x3, x4, dim=1))


class Darknet:
    def __init__(self, w, r, d):
        self.b1 = [
            Conv_Block(c1=3, c2=int(64 * w), kernel_size=3, stride=2, padding=1),
            Conv_Block(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1),
        ]
        self.b2 = [
            C2f(c1=int(128 * w), c2=int(128 * w), n=round(3 * d), shortcut=True),
            Conv_Block(int(128 * w), int(256 * w), 3, 2, 1),
            C2f(int(256 * w), int(256 * w), round(6 * d), True),
        ]
        self.b3 = [
            Conv_Block(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1),
            C2f(int(512 * w), int(512 * w), round(6 * d), True),
        ]
        self.b4 = [
            Conv_Block(
                int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1
            ),
            C2f(int(512 * w * r), int(512 * w * r), round(3 * d), True),
        ]
        self.b5 = [SPPF(int(512 * w * r), int(512 * w * r), 5)]

    def return_modules(self):
        return [*self.b1, *self.b2, *self.b3, *self.b4, *self.b5]

    def __call__(self, x):
        x1 = x.sequential(self.b1)
        x2 = x1.sequential(self.b2)
        x3 = x2.sequential(self.b3)
        x4 = x3.sequential(self.b4)
        x5 = x4.sequential(self.b5)
        return (x2, x3, x5)


class Yolov8NECK:
    def __init__(self, w, r, d):  # width_multiple, ratio_multiple, depth_multiple
        self.up = Upsample(2, mode="nearest")
        self.n1 = C2f(
            c1=int(512 * w * (1 + r)), c2=int(512 * w), n=round(3 * d), shortcut=False
        )
        self.n2 = C2f(c1=int(768 * w), c2=int(256 * w), n=round(3 * d), shortcut=False)
        self.n3 = Conv_Block(
            c1=int(256 * w), c2=int(256 * w), kernel_size=3, stride=2, padding=1
        )
        self.n4 = C2f(c1=int(768 * w), c2=int(512 * w), n=round(3 * d), shortcut=False)
        self.n5 = Conv_Block(
            c1=int(512 * w), c2=int(512 * w), kernel_size=3, stride=2, padding=1
        )
        self.n6 = C2f(
            c1=int(512 * w * (1 + r)),
            c2=int(512 * w * r),
            n=round(3 * d),
            shortcut=False,
        )

    def return_modules(self):
        return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

    def __call__(self, p3, p4, p5):
        x = self.n1(self.up(p5).cat(p4, dim=1))
        head_1 = self.n2(self.up(x).cat(p3, dim=1))
        head_2 = self.n4(self.n3(head_1).cat(x, dim=1))
        head_3 = self.n6(self.n5(head_2).cat(p5, dim=1))
        return [head_1, head_2, head_3]
