import math
from typing import Tuple
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.nn import Conv2d, Linear

class BatchNorm2d(nn.BatchNorm2d):
  def __init__(self, dim:int, eps=1e-5): super().__init__(dim, eps)
  def __call__(self, x:Tensor) -> Tensor: return super().__call__(x.float()).cast(dtypes.default_float)

def make_divisible(x: float, divisible_by: int) -> int:
  return int(x + divisible_by / 2) // divisible_by * divisible_by

def hardsigmoid(x: Tensor) -> Tensor: return (x + 3).relu6() / 6

class SE:
  def __init__(self, dim:int, se_ratio=0.25):
    reduced = make_divisible(dim * se_ratio, 4)
    self.cv1 = Conv2d(dim, reduced, kernel_size=1, bias=False)
    self.cv2 = Conv2d(reduced, dim, kernel_size=1, bias=False)
  def __call__(self, x: Tensor):
    xx = x.mean((2, 3), keepdim=True)
    xx = self.cv1(xx).relu()
    xx = hardsigmoid(self.cv2(xx))
    return x * xx

def upsample(x: Tensor, scale: int):
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

class GhostModuleV2:
  def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, attn=False):
    self.oup, self.attn = oup, attn
    init_channels = math.ceil(oup / ratio)
    new_channels = init_channels * (ratio - 1)

    self.primary_conv = [
      Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
      BatchNorm2d(init_channels),
    ]
    if relu: self.primary_conv.append(Tensor.relu)
    self.cheap_operation = [
      Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
      BatchNorm2d(new_channels),
    ]
    if relu: self.cheap_operation.append(Tensor.relu)

    if attn:
      self.short_conv = [
        Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
        BatchNorm2d(oup),
        Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
        BatchNorm2d(oup),
        Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
        BatchNorm2d(oup),
      ]
  def __call__(self, x:Tensor) -> Tensor:
    if self.attn: res = x.avg_pool2d(kernel_size=2, stride=2).sequential(self.short_conv)
    x1 = x.sequential(self.primary_conv)
    x2 = x1.sequential(self.cheap_operation)
    out = x1.cat(x2, dim=1)
    if self.attn: out = out * upsample(res.sigmoid(), 2)
    return out

class GhostBottleneckV2:
  def __init__(self, layer_id, cin, cmid, cout, dw_size=3, stride=1, se_ratio=0.):
    self.stride = stride

    self.ghost1 = GhostModuleV2(cin, cmid, relu=True, attn=layer_id > 1)

    if stride > 1:
      self.conv_dw = Conv2d(cmid, cmid, dw_size, stride, dw_size//2, groups=cmid, bias=False)
      self.bn_dw = BatchNorm2d(cmid)

    if se_ratio > 0:
      self.se = SE(cmid, se_ratio)

    self.ghost2 = GhostModuleV2(cmid, cout, relu=False)

    if cin == cout and stride == 1:
      self.shortcut = []
    else:
      self.shortcut = [
        Conv2d(cin, cin, dw_size, stride=stride, padding=dw_size//2, groups=cin, bias=False),
        BatchNorm2d(cin),
        Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False),
        BatchNorm2d(cout),
      ]
  def __call__(self, x:Tensor) -> Tensor:
    res = x
    x = self.ghost1(x)
    if self.stride > 1:
      x = self.bn_dw(self.conv_dw(x))
    if hasattr(self, "se"): x = self.se(x)
    x = self.ghost2(x)
    x = x + res.sequential(self.shortcut)
    return x

class ConvBnAct:
  def __init__(self, cin, cout, kernel_size, stride=1):
    self.conv = Conv2d(cin, cout, kernel_size, stride, kernel_size//2, bias=False)
    self.bn1 = BatchNorm2d(cout)
  def __call__(self, x:Tensor) -> Tensor: return self.bn1(self.conv(x)).relu()

class GhostNetV2:
  def __init__(self):
    self.conv_stem = Conv2d(3, 16, 3, 2, 1, bias=False)
    self.bn1 = BatchNorm2d(16)

    self.blocks = [
      [GhostBottleneckV2(0, cin=16, cmid=16, cout=16, dw_size=3, stride=1, se_ratio=0.)],
      [GhostBottleneckV2(1, cin=16, cmid=48, cout=24, dw_size=3, stride=2, se_ratio=0.)],
      [GhostBottleneckV2(2, cin=24, cmid=72, cout=24, dw_size=3, stride=1, se_ratio=0.)],
      [GhostBottleneckV2(3, cin=24, cmid=72, cout=40, dw_size=5, stride=2, se_ratio=0.25)],
      [GhostBottleneckV2(4, cin=40, cmid=120, cout=40, dw_size=5, stride=1, se_ratio=0.25)],
      [GhostBottleneckV2(5, cin=40, cmid=240, cout=80, dw_size=3, stride=2, se_ratio=0.)],
      [
        GhostBottleneckV2(6, cin=80, cmid=200, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(7, cin=80, cmid=184, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(8, cin=80, cmid=184, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(9, cin=80, cmid=480, cout=112, dw_size=3, stride=1, se_ratio=0.25),
        GhostBottleneckV2(10, cin=112, cmid=672, cout=112, dw_size=3, stride=1, se_ratio=0.25),
      ],
      [GhostBottleneckV2(11, cin=112, cmid=672, cout=160, dw_size=5, stride=2, se_ratio=0.25)],
      [
        GhostBottleneckV2(12, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.),
        GhostBottleneckV2(13, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.25),
        GhostBottleneckV2(14, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.),
        GhostBottleneckV2(15, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.25),
      ],
      [ConvBnAct(160, 960, 1)],
    ]
  def __call__(self, x: Tensor) -> Tensor:
    x = self.bn1(self.conv_stem(x)).relu()
    for block in self.blocks:
      x = x.sequential(block)
      print(x.shape)
    return x

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad import GlobalCounters

  model = GhostNetV2()
  print("model params:", sum(x.numel() for x in get_parameters(model)) / 1e6)
  model(Tensor.zeros(1, 3, 128, 256)).realize()

  GlobalCounters.reset()

  model(Tensor.zeros(1, 3, 128, 256)).realize()
  print("model params:", sum(x.numel() for x in get_parameters(model)) / 1e6)
