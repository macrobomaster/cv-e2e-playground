import glob, math, random, sys, signal, time, multiprocessing
from multiprocessing import Queue, Process

from tinygrad import dtypes, Tensor, GlobalCounters
from tinygrad import TinyJit
from tinygrad.nn.optim import SGD, LAMB, AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.helpers import Context
from tqdm import trange
import wandb
import cv2

from model import Model
from main import BASE_PATH

BS = 32
WARMUP_STEPS = 1000
WARMPUP_LR = 0.0001
START_LR = 0.005
END_LR = 0.0001
STEPS = 100000

def pseudo_huber_loss(pred: Tensor, y: Tensor, delta: float = 1.0): return ((delta*delta) * ((1 + ((pred - y) / delta).square()).sqrt() - 1)).mean()
def loss_fn(pred: tuple[Tensor, Tensor], y: Tensor):
  obj_loss = pred[0][:, 0, 0].binary_crossentropy_logits(y[:, 0])
  # x_loss = pseudo_huber_loss(pred[1][:, 0, 0], y[:, 1])
  x_loss = (pred[1][:, 0, 0] - y[:, 1]).abs().mul(pred[0][:, 0, 0].sigmoid() + y[:, 0] + 0.4).mean()
  # y_loss = pseudo_huber_loss(pred[1][:, 0, 1], y[:, 2])
  y_loss = (pred[1][:, 0, 1] - y[:, 2]).abs().mul(pred[0][:, 0, 0].sigmoid() + y[:, 0] + 0.4).mean()

  return obj_loss + x_loss + y_loss

@TinyJit
def train_step(x, y, lr):
  pred = model(x)
  loss = loss_fn(pred, y)

  optim.lr.assign(lr+1-1)
  optim.zero_grad()
  loss.backward()
  optim.step()

  return loss.float().realize()

warming_up = True
def get_lr(i: int) -> float:
  global warming_up
  if warming_up:
    lr = START_LR * (i / WARMUP_STEPS) + WARMPUP_LR * (1 - i / WARMUP_STEPS)
    if i >= WARMUP_STEPS: warming_up = False
  else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((i - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)) * math.pi))
  return lr

preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.png"))
def load_single_file(file):
  img = cv2.imread(file)
  # read the annotation file
  annotation_file = file.replace(".png", ".txt")
  with open(annotation_file, "r") as f:
    detected, x, y, color = f.readline().split(" ")
    detected, color = int(detected), int(color)
    x, y = float(x), float(y)
  return img, (detected, x, y, color)

def minibatch_iterator(q: Queue):
  pool = multiprocessing.Pool(4)
  while True:
    random.shuffle(preprocessed_train_files)
    for i in range(0, len(preprocessed_train_files) - BS, BS):
      batched = pool.map(load_single_file, preprocessed_train_files[i : i + BS])
      x_b, y_b = zip(*batched)
      q.put((list(x_b), list(y_b)))

class ModelEMA:
  def __init__(self, model):
    self.model = Model()
    for ep, p in zip(get_state_dict(self.model).values(), get_state_dict(model).values()):
      ep.requires_grad = False
      ep.assign(p)

  @TinyJit
  def update(self, net, alpha):
    for ep, p in zip(get_state_dict(self.model).values(), get_state_dict(net).values()):
      ep.assign(alpha * ep.detach() + (1 - alpha) * p.detach()).realize()

if __name__ == "__main__":
  Tensor.no_grad = False
  Tensor.training = True
  # dtypes.default_float = dtypes.float16

  wandb.init(project="mrm_e2e_playground")
  wandb.config.update({
    "warmup_steps": WARMUP_STEPS,
    "start_lr": START_LR,
    "end_lr": END_LR,
    "steps": STEPS,
  })

  model = Model()

  sn_state_dict = safe_load("./weights/shufflenetv2.safetensors")
  load_state_dict(model.backbone, sn_state_dict)

  # state_dict = safe_load(str(BASE_PATH / "model_0.safetensors"))
  # load_state_dict(model, state_dict)

  model_ema = ModelEMA(model)

  parameters = []
  for key, value in get_state_dict(model).items():
    # if "backbone" in key: continue
    parameters.append(value)
  optim = SGD(parameters, momentum=0.9, nesterov=True, weight_decay=1e-5)
  # optim = AdamW(parameters, wd=1e-4)

  # start batch iterator in a separate process
  bi_queue = Queue(4)
  bi = Process(target=minibatch_iterator, args=(bi_queue,))
  bi.start()

  def sigint_handler(*_):
    print("SIGINT received, killing batch iterator")
    bi.terminate()
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  for step in (t := trange(STEPS)):
    with Context(BEAM=0 if step == 0 else 4):
      GlobalCounters.reset()
      st = time.perf_counter()

      # train one step
      new_lr = get_lr(step)
      if step == 0:
        x, y = bi_queue.get()
        x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)
      loss = train_step(x, y, Tensor([new_lr], dtype=dtypes.default_float))
      x, y = bi_queue.get()
      x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)

      # # update EMA
      # if step >= 400 and step % 5 == 0: model_ema.update(model, Tensor([0.998]))
      #
      # # sema
      # if step >= 600 and step % 200 == 0:
      #   for p, ep in zip(get_state_dict(model).values(), get_state_dict(model_ema.model).values()):
      #     p.assign(ep.detach()).realize()
      et = time.perf_counter() - st

      loss, lr = loss.item(), new_lr
      t.set_description(f"loss: {loss:6.6f}, lr: {lr:12.12f}")
      wandb.log({
        "loss": loss,
        "lr": lr,
        "gflop": GlobalCounters.global_ops / et / 1e9,
        "gb": GlobalCounters.global_mem / et / 1e9,
      })

      if step % 10000 == 0 and step > 0: safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{step}.safetensors"))
  safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))

  bi.terminate()
