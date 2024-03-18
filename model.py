from tinygrad.nn import Conv2d, Linear
from tinygrad import Tensor, dtypes

from backbones.shufflenet import ShuffleNetV2, BatchNorm2d
from backbones.ghostnet import GhostNetV2

def hardsigmoid(x: Tensor) -> Tensor: return (x + 3).relu6() / 6

class FFNBlock:
  def __init__(self, dim, e=2):
    self.l1 = Linear(dim, dim * e)
    self.l2 = Linear(dim * e, dim)
  def __call__(self, x:Tensor):
    x_ = self.l1(x).relu()
    return (x + self.l2(x_)).relu()

class FFN:
  def __init__(self, dim, blocks=2):
    self.blocks = [FFNBlock(dim) for _ in range(blocks)]
  def __call__(self, x:Tensor): return x.sequential(self.blocks)

class SE:
  def __init__(self, dim:int):
    self.cv1 = Conv2d(dim, dim//16, kernel_size=1, bias=False)
    self.cv2 = Conv2d(dim//16, dim, kernel_size=1, bias=False)
  def __call__(self, x: Tensor):
    xx = x.mean((2, 3), keepdim=True)
    xx = self.cv1(xx).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class ObjHead:
  def __init__(self, in_dim, dim, num_outputs):
    self.proj = Linear(in_dim, dim)
    self.l1 = Linear(dim, dim)
    self.l2 = Linear(dim, dim)
    self.out = Linear(dim, num_outputs)
  def __call__(self, x:Tensor):
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
  def __call__(self, x:Tensor):
    x = self.proj(x).relu()
    x_ = self.l1(x).relu()
    x = (x + self.l2(x_)).relu()
    return self.out(x).sigmoid().reshape(x.shape[0], -1, 2)

class Neck:
  def __init__(self, cin:int, dim:int):
    self.se = SE(cin)
    self.conv = Conv2d(cin, 32, 1, 1, 0, bias=False)
    self.bn = BatchNorm2d(32)
    self.proj = Linear(1024, dim)
    self.ffn = FFN(dim, blocks=2)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.se(x)
    x = self.bn(self.conv(x)).relu()
    x = x.flatten(1)
    x = self.proj(x).relu()
    return self.ffn(x)

class Head:
  def __init__(self, dim:int, num_outputs:int):
    self.obj = ObjHead(dim, 64, num_outputs)
    self.pos = PosHead(dim, 64, num_outputs)
  def __call__(self, x:Tensor) -> Tuple[Tensor, Tensor]: return self.obj(x), self.pos(x)

class Model:
  def __init__(self):
    self.backbone = ShuffleNetV2()
    # self.backbone = GhostNetV2()
    self.neck = Neck(1024, 256)
    self.head = Head(256, num_outputs=1)

  def __call__(self, img: Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    # backbone
    x = self.backbone(img)

    # neck
    x = self.neck(x)

    # head
    obj, pos = self.head(x)

    # cast to correct output type
    if not Tensor.training: obj, pos = obj.cast(dtypes.default_float), pos.cast(dtypes.default_float)
    return obj, pos

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
  print("model params:", sum(x.numel() for x in get_parameters(model)) / 1e6)
