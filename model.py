from tinygrad.nn import Conv2d, Linear, BatchNorm2d
from tinygrad import Tensor, dtypes

from shufflenet import ShuffleNetV2
from backbone import Backbone

class ObjHead:
  def __init__(self, in_dim, dim, num_outputs):
    self.proj = Linear(in_dim, dim)
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, dim)
    self.out = Linear(dim, num_outputs)

  def __call__(self, x: Tensor):
    x = self.proj(x).relu()
    x_ = self.l1(x).relu()
    x = (x + self.l2(x_)).relu()
    if Tensor.training: return self.out(x).reshape(x.shape[0], -1, 1)
    else: return self.out(x).sigmoid().reshape(x.shape[0], -1, 1)

class PosHead:
  def __init__(self, in_dim, dim, num_outputs):
    self.proj = Linear(in_dim, dim)
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, dim)
    self.out = Linear(dim, num_outputs * 2)

  def __call__(self, x: Tensor):
    x = self.proj(x).relu()
    x_ = self.l1(x).relu()
    x = (x + self.l2(x_)).relu()
    return self.out(x).sigmoid().reshape(x.shape[0], -1, 2)

class FFNBlock:
  def __init__(self, dim, e=2):
    self.l1 = Linear(dim, dim * e)
    self.l2 = Linear(dim * e, dim)
  def __call__(self, x: Tensor):
    x_ = self.l1(x).relu()
    return (x + self.l2(x_)).relu()

class FFN:
  def __init__(self, dim, blocks=2):
    self.blocks = [FFNBlock(dim) for _ in range(blocks)]
  def __call__(self, x: Tensor): return x.sequential(self.blocks)

class Model:
  def __init__(self):
    # self.backbone = Backbone()
    self.backbone = ShuffleNetV2()
    self.head_conv = Conv2d(1024, 64, 1, 1, 0)
    self.proj = Linear(2048, 512)
    self.ffn = FFN(512, blocks=2)
    self.obj_head = ObjHead(512, 64, num_outputs=1)
    self.pos_head = PosHead(512, 64, num_outputs=1)

  def __call__(self, img: Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    # backbone
    x = self.backbone(img)

    # head transform
    x = self.head_conv(x)
    x = x.flatten(1)
    x = self.proj(x).relu()
    x = self.ffn(x)

    # heads
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
  x_obj, x_pos = model(Tensor.zeros(1, 128, 256, 3))
  x_obj.realize()
  x_pos.realize()

  GlobalCounters.reset()

  x_obj, x_pos = model(Tensor.zeros(1, 128, 256, 3))
  x_obj.realize()
  x_pos.realize()
