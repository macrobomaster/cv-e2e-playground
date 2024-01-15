import glob, math, random, sys, signal
import multiprocessing
from multiprocessing import Queue, Process

from tinygrad import dtypes, Tensor, GlobalCounters, Device
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD, LAMB, AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.helpers import Context, all_int
from tqdm import trange
import wandb
import cv2

from model import Model
from main import BASE_PATH

BS = 2
WARMUP_STEPS = 1000
WARMPUP_LR = 0.0001
START_LR = 0.001
END_LR = 0.0005
STEPS = 10000

def cartesian_product(x: Tensor, y: Tensor):
  assert x.ndim == 1 and y.ndim == 1
  x_len, y_len = x.shape[0], y.shape[0]
  x = x.unsqueeze(1).expand(-1, y_len).reshape(-1, 1)
  y = y.unsqueeze(0).expand(x_len, -1).reshape(-1, 1)
  return x.cat(y, dim=1)

def pairwise_distance(x: Tensor, y: Tensor):
  x, y = x.unsqueeze(1), y.unsqueeze(0)
  return (x - y).pow(2).sum(2).sqrt()

all_img_locations = None
def weighted_hausdorff_distance(pred: Tensor, gt: Tensor):
  global all_img_locations
  if all_img_locations is None:
    all_img_locations = cartesian_product(Tensor.arange(pred.shape[1]), Tensor.arange(pred.shape[2]))

  assert pred.ndim == 3, "pred should be BxHxW"
  assert all_int(pred.shape), "symbolic shapes not supported"
  term_1s, term_2s = [], []
  for b in range(pred.shape[0]):
    pred_b, gt_b = pred[b], gt[b]

    # pairwise distance between all possible locations and the ground truth
    d_matrix = pairwise_distance(all_img_locations, gt_b)

    # probability map
    p = pred_b.flatten()
    n_est_pts = p.sum()
    p_replicated = p.reshape(-1, 1).repeat((1, gt_b.shape[0]))

    # weighted hausdorff distance
    term_1s.append((1 / (n_est_pts + 1e-6)) * (p * d_matrix.min(1)[0]).sum())
    weighted_d_matrix = (1 - p_replicated) * (pred.shape[1]**2 + pred.shape[2]**2)**0.5 + p_replicated * d_matrix
    minn = ((weighted_d_matrix + 1e-6)**-3).mean(0) ** (1/-3)
    term_2s.append(minn.mean())
  return Tensor.stack(term_1s).mean() + Tensor.stack(term_2s).mean()

def pseudo_huber_loss(pred: Tensor, y: Tensor | float, delta: float = 1.0): return ((delta*delta) * ((1 + ((pred - y) / delta).square()).sqrt() - 1)).mean()

def loss_fn(pred: tuple[Tensor, Tensor], y: Tensor):
  whd_loss = weighted_hausdorff_distance(pred[0], y)
  ph_loss = pseudo_huber_loss(pred[1], 1)
  return whd_loss + ph_loss

@TinyJit
def train_step(x, y, lr):
  pred = model(x)
  loss = loss_fn(pred, y)

  optim.lr.assign(lr+1-1)
  optim.zero_grad()
  loss.backward()

  # calculate grad norm
  grad_norm = Tensor([0.], dtype=dtypes.float32)
  for p in get_parameters(model):
    if p.grad is not None:
      grad_norm.assign(grad_norm + p.grad.detach().pow(2).sum())

  optim.step()

  return loss.realize(), grad_norm.realize()

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
    _, x, y, _ = f.readline().split(" ")
    x, y = float(x), float(y)
  return img, [(x, y)]

def minibatch_iterator(q: Queue):
  pool = multiprocessing.Pool(4)
  while True:
    random.shuffle(preprocessed_train_files)
    for i in range(0, len(preprocessed_train_files) - BS, BS):
      batched = pool.map(load_single_file, preprocessed_train_files[i : i + BS])
      x_b, y_b = zip(*batched)
      q.put((list(x_b), list(y_b)))

if __name__ == "__main__":
  Tensor.no_grad = False
  Tensor.training = True
  dtypes.default_float = dtypes.float32

  wandb.init(project="mrm_e2e_playground")
  wandb.config.update({
    "warmup_steps": WARMUP_STEPS,
    "start_lr": START_LR,
    "end_lr": END_LR,
    "steps": STEPS,
  })

  model = Model()

  # sn_state_dict = safe_load("./weights/shufflenetv2.safetensors")
  # load_state_dict(model.backbone, sn_state_dict)

  # state_dict = safe_load(str(BASE_PATH / "model_0.safetensors"))
  # load_state_dict(model, state_dict)

  parameters = []
  for key, value in get_state_dict(model).items():
    # if "backbone" in key: continue
    parameters.append(value)
  optim = SGD(parameters, momentum=0.9, weight_decay=1e-4)
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

  with Context(BEAM=0):
    for step in (t := trange(STEPS)):
      GlobalCounters.reset()
      new_lr = get_lr(step)
      if step == 0:
        x, y = bi_queue.get()
        x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)
      loss, grad_norm = train_step(x, y, Tensor([new_lr], dtype=dtypes.default_float))
      x, y = bi_queue.get()
      x, y = Tensor(x, dtype=dtypes.uint8), Tensor(y, dtype=dtypes.default_float)
      loss, grad_norm, lr = loss.item(), grad_norm.item(), optim.lr.item()
      t.set_description(f"loss: {loss:6.6f}, grad_norm: {grad_norm:6.6f}, lr: {lr:12.12f}")
      wandb.log({
        "loss": loss,
        "grad_norm": grad_norm,
        "lr": lr,
      })

      if step % 10000 == 0 and step > 0: safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_radar_{step}.safetensors"))
    safe_save(get_state_dict(model), str(BASE_PATH / f"model_radar.safetensors"))

  bi.terminate()
