import math, time

from tinygrad import Device, dtypes, Tensor, GlobalCounters
from tinygrad import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.helpers import Context
from tqdm import tqdm
import wandb

from model import Model
from main import BASE_PATH
from dataloader import batch_load, preprocessed_train_files

BS = 256
WARMUP_STEPS = 100
WARMPUP_LR = 0.0001
START_LR = 0.005
END_LR = 0.0001
EPOCHS = 10
STEPS_PER_EPOCH = len(preprocessed_train_files)//BS

def pseudo_huber_loss(pred: Tensor, y: Tensor, delta: float = 1.0): return ((delta*delta) * ((1 + ((pred - y) / delta).square()).sqrt() - 1)).mean()
def loss_fn(pred: tuple[Tensor, Tensor], y: Tensor):
  obj_loss = pred[0][:, 0, 0].binary_crossentropy_logits(y[:, 0])
  # x_loss = pseudo_huber_loss(pred[1][:, 0, 0], y[:, 1])
  # y_loss = pseudo_huber_loss(pred[1][:, 0, 1], y[:, 2])
  leaky_gate = pred[0][:, 0, 0].sigmoid() + y[:, 0] + 0.4
  x_loss = (pred[1][:, 0, 0] - y[:, 1]).abs().mul(leaky_gate).mean()
  y_loss = (pred[1][:, 0, 1] - y[:, 2]).abs().mul(leaky_gate).mean()

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
def get_lr(step:int) -> float:
  global warming_up
  if warming_up:
    lr = START_LR * (step / WARMUP_STEPS) + WARMPUP_LR * (1 - step / WARMUP_STEPS)
    if step >= WARMUP_STEPS: warming_up = False
  else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((step - WARMUP_STEPS) / ((EPOCHS * STEPS_PER_EPOCH) - WARMUP_STEPS)) * math.pi))
  return lr

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
    "warmup_lr": WARMPUP_LR,
    "start_lr": START_LR,
    "end_lr": END_LR,
    "epochs": EPOCHS,
    "bs": BS,
    "steps_per_epoch": STEPS_PER_EPOCH,
  })

  model = Model()

  sn_state_dict = safe_load("./weights/shufflenetv2.safetensors")
  load_state_dict(model.backbone, sn_state_dict)

  # state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  # load_state_dict(model, state_dict)

  model_ema = ModelEMA(model)

  parameters = get_parameters(model)
  optim = SGD(parameters, momentum=0.9, nesterov=True, weight_decay=1e-5)

  def single_batch(iter):
    x, y, c = next(iter)
    return x.to(Device.DEFAULT), y.to(Device.DEFAULT), c

  steps = 0
  for epoch in range(EPOCHS):
    batch_iter = iter(tqdm(batch_load(BS), total=STEPS_PER_EPOCH, desc=f"epoch {epoch}"))
    i, proc = 0, single_batch(batch_iter)
    while proc is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      lr = get_lr(steps)
      loss = train_step(proc[0], proc[1], Tensor([lr], dtype=dtypes.default_float))
      pt = time.perf_counter()

      try: next_proc = single_batch(batch_iter)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss = loss.item()
      at = time.perf_counter()

      tqdm.write(
        f"{i:5} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{loss:11.6f} loss, {lr:.6f} lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      wandb.log({
        "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
        "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
        "loss": loss, "lr": lr,
        "gb": GlobalCounters.mem_used / 1e9, "gbps": GlobalCounters.mem_used * 1e-9 / (at - st), "gflops": GlobalCounters.global_ops * 1e-9 / (at - st)
      })

      proc, next_proc = next_proc, None
      i += 1
      steps += 1

      # # update EMA
      # if step >= 400 and step % 5 == 0: model_ema.update(model, Tensor([0.998]))
      #
      # # sema
      # if step >= 600 and step % 200 == 0:
      #   for p, ep in zip(get_state_dict(model).values(), get_state_dict(model_ema.model).values()):
      #     p.assign(ep.detach()).realize()

    if epoch != EPOCHS - 1: safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))
  safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))
