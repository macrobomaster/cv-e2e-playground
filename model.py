from itertools import chain

from tinygrad.nn import Conv2d, Linear, BatchNorm2d
from tinygrad import Tensor, dtypes

from shufflenet import ShuffleNetV2

def upsample(x: Tensor, scale_factor: int):
  assert len(x.shape) > 2 and len(x.shape) <= 5
  (b, c), _lens = x.shape[:2], len(x.shape[2:])
  tmp = (x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [scale_factor] * _lens)).reshape(list(x.shape) + [scale_factor] * _lens)
  return tmp.permute([0, 1] + list(chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)]))).reshape([b, c] + [x * scale_factor for x in x.shape[2:]])

class DFCAttention:
  def __init__(self, dim, *, attention_size=5):
    self.cv = Conv2d(dim, dim, kernel_size=1, bias=False)
    self.norm = BatchNorm2d(dim)

    # horizontal fc
    self.hcv = Conv2d(dim, dim, kernel_size=(1, attention_size), padding=(0, attention_size//2), groups=dim, bias=False)
    self.hnorm = BatchNorm2d(dim)
    # vertical fc
    self.vcv = Conv2d(dim, dim, kernel_size=(attention_size, 1), padding=(attention_size//2, 0), groups=dim, bias=False)
    self.vnorm = BatchNorm2d(dim)

  def __call__(self, x: Tensor) -> Tensor:
    assert x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0, f"attention input must be divisible by 2, got {x.shape}"
    # downsample
    xx = x.avg_pool2d(2)
    # attention map
    xx = self.norm(self.cv(xx).float()).cast(dtypes.default_float)
    xx = self.hnorm(self.hcv(xx).float()).cast(dtypes.default_float)
    xx = self.vnorm(self.vcv(xx).float()).cast(dtypes.default_float)
    xx = xx.sigmoid()
    # upsample
    return upsample(xx, 2)

class EncoderBlock:
  def __init__(self, dim):
    self.attention = DFCAttention(dim)
    self.norm1 = BatchNorm2d(dim)
    self.cv1 = Conv2d(dim, dim, kernel_size=1, bias=False)
    self.cv2 = Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
    self.norm2 = BatchNorm2d(dim)

  def __call__(self, x: Tensor):
    # attention
    xx = x * self.attention(x)
    x = x + xx
    x = self.norm1(x.float()).cast(dtypes.default_float)
    # feedforward
    xx = self.cv1(x).mish()
    xx = self.cv2(xx).mish()
    x = x + xx
    return self.norm2(x.float()).cast(dtypes.default_float)

class SEBlock:
  def __init__(self, dim):
    self.cv1 = Conv2d(dim, dim//16, kernel_size=1, bias=False)
    self.cv2 = Conv2d(dim//16, dim, kernel_size=1, bias=False)

  def __call__(self, x: Tensor):
    xx = x.avg_pool2d(x.shape[-1])
    xx = self.cv1(xx).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class Encoder:
  def __init__(self, dim):
    self.dim = dim

    self.encoders = [EncoderBlock(dim) for _ in range(2)]
    self.se = SEBlock(dim)
    self.cv_out = Conv2d(dim, 32, kernel_size=1, bias=False)

  def __call__(self, x: Tensor):
    x = x.sequential(self.encoders)
    x = self.se(x).mish()
    x = self.cv_out(x).flatten(1)
    return x

class ObjHead:
  def __init__(self, dim, num_outputs):
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, num_outputs)
  def __call__(self, x: Tensor):
    x = self.l1(x).mish()
    if Tensor.training: return self.l2(x).reshape(x.shape[0], -1, 1)
    else: return self.l2(x).sigmoid().reshape(x.shape[0], -1, 1)

class PosHead:
  def __init__(self, dim, num_outputs):
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, dim)
    self.l3 = Linear(dim, num_outputs * 2)

  def __call__(self, x: Tensor):
    x = self.l1(x).mish()
    x = self.l2(x).mish()
    return self.l3(x).sigmoid().reshape(x.shape[0], -1, 2)

class Model:
  def __init__(self):
    self.backbone = ShuffleNetV2()

    self.input_conv = Conv2d(1024, 512, kernel_size=3, bias=False)
    self.enc = Encoder(512)

    self.proj = Linear(2048, 512)
    self.obj_head = ObjHead(512, 1)
    self.pos_head = PosHead(512, 1)

  def __call__(self, img: Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    # backbone
    x = self.backbone(img)

    # encoder-decoder
    x = self.input_conv(x)
    x = self.enc(x)

    # heads
    x = self.proj(x).mish()
    x_obj = self.obj_head(x)
    x_pos = self.pos_head(x)

    # cast to correct output type
    if not Tensor.training: x_obj, x_pos = x_obj.cast(dtypes.default_float), x_pos.cast(dtypes.default_float)
    return x_obj, x_pos

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad import GlobalCounters

  model = Model()
  print("model params:", sum(x.numel() for x in get_parameters(model)) / 1e6)
  x_obj, x_pos = model(Tensor.zeros(1, 320, 320, 3))
  x_obj.realize()
  x_pos.realize()

  GlobalCounters.reset()

  x_obj, x_pos = model(Tensor.zeros(1, 320, 320, 3))
  x_obj.realize()
  x_pos.realize()
