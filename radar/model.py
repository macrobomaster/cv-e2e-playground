from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv2d, ConvTranspose2d, BatchNorm2d, Linear

def upsample(x: Tensor, scale: int):
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

class ConvBlock:
  def __init__(self, c_in, c_out):
    self.cv1 = Conv2d(c_in, c_out, 3, padding=1)
    self.norm1 = BatchNorm2d(c_out)
    self.cv2 = Conv2d(c_out, c_out, 3, padding=1)
    self.norm2 = BatchNorm2d(c_out)

  def __call__(self, x: Tensor):
    x = self.norm1(self.cv1(x).float()).cast(dtypes.default_float).mish()
    x = self.norm2(self.cv2(x).float()).cast(dtypes.default_float).mish()
    return x

class DownBlock:
  def __init__(self, c_in, c_out):
    self.cv = ConvBlock(c_in, c_out)

  def __call__(self, x: Tensor):
    x = x.max_pool2d(2)
    x = self.cv(x)
    return x

class UpBlock:
  def __init__(self, c_in, c_out):
    self.cv = ConvBlock(c_in, c_out)

  def __call__(self, x: Tensor):
    x = upsample(x, 2)
    x = self.cv(x)
    return x

class Model:
  def __init__(self):
    self.input_conv = ConvBlock(3, 32)

    self.down1 = DownBlock(32, 32)
    self.down2 = DownBlock(32, 32)
    self.down3 = DownBlock(32, 32)

    self.cntr = ConvBlock(32, 32)

    self.up1 = UpBlock(64, 32)
    self.up2 = UpBlock(64, 32)
    self.up3 = UpBlock(64, 32)

    self.output_conv = Conv2d(32, 1, 1)

    self.l1 = Linear(32, 64)
    self.l2 = Linear(32, 64)
    self.l3 = Linear(128, 1)

  def __call__(self, img: Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    x = self.input_conv(img)

    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2)

    x4 = self.cntr(x3)

    x = self.up1(x4.cat(x3, dim=1))
    x = self.up2(x.cat(x2, dim=1))
    x = self.up3(x.cat(x1, dim=1))

    xx = self.l1(x3.mean((2, 3))).mish()
    xxx = self.l2(x.mean((2, 3))).mish()
    xx = self.l3(xx.cat(xxx, dim=1)).relu()

    x = self.output_conv(x).squeeze(1).sigmoid()

    return x, xx

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad import GlobalCounters

  model = Model()
  print("model params:", sum(x.numel() for x in get_parameters(model)) / 1e6)
  x, xx = model(Tensor.zeros(1, 320, 320, 3))
  x.realize()
  xx.realize()

  GlobalCounters.reset()

  x, xx = model(Tensor.zeros(1, 320, 320, 3))
  x.realize()
  xx.realize()
