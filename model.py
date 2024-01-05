from itertools import chain

from tinygrad.nn import Conv2d, Linear, BatchNorm2d
from tinygrad.tensor import Tensor

from shufflenet import ShuffleNetV2

class EncoderDecoder:
  def __init__(self, dim):
    self.cv1 = Conv2d(dim, dim // 2, kernel_size=3, bias=False)
    self.norm1 = BatchNorm2d(dim // 2)
    self.cv2 = Conv2d(dim // 2, dim // 2, kernel_size=3, bias=False)
    self.norm2 = BatchNorm2d(dim // 2)
    self.cv3 = Conv2d(dim // 2, dim, kernel_size=3, bias=False)
    self.norm3 = BatchNorm2d(dim)

  def __call__(self, x: Tensor):
    x = self.norm1(self.cv1(xx := x)).gelu()
    x = self.norm2(self.cv2(x)).gelu()
    x = self.norm3(self.cv3(x) + xx).gelu()
    x = x.mean((2, 3)).reshape(x.shape[0], -1)
    return x

class ObjHead:
  def __init__(self, dim, num_outputs):
    self.l1 = Linear(dim, num_outputs)
  def __call__(self, x: Tensor): return self.l1(x).reshape(x.shape[0], -1, 1)

class PosHead:
  def __init__(self, dim, num_outputs):
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, dim)
    self.l3 = Linear(dim, num_outputs * 2)

  def __call__(self, x: Tensor):
    x = self.l1(x).gelu()
    x = self.l2(x).gelu()
    return self.l3(x).sigmoid().reshape(x.shape[0], -1, 2)

def upsample(x: Tensor, scale_factor: int):
  assert len(x.shape) > 2 and len(x.shape) <= 5
  (b, c), _lens = x.shape[:2], len(x.shape[2:])
  tmp = (x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [scale_factor] * _lens)).reshape(list(x.shape) + [scale_factor] * _lens)
  return tmp.permute([0, 1] + list(chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)]))).reshape([b, c] + [x * scale_factor for x in x.shape[2:]])

class Model:
  def __init__(self):
    self.backbone = ShuffleNetV2()

    self.input_conv = Conv2d(1024, 512, kernel_size=1, bias=False)
    self.encdec = EncoderDecoder(512)

    self.obj_head = ObjHead(512, 1)
    self.pos_head = PosHead(512, 1)

  def __call__(self, img: Tensor):
    # image normalization
    img = img.permute(0, 3, 1, 2).float() / 255

    x = self.backbone(img)
    x = self.input_conv(x)
    x = self.encdec(x)

    x_obj = self.obj_head(x)
    x_pos = self.pos_head(x)
    return x_obj, x_pos
