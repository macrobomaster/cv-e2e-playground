import glob
import math
import random
from multiprocessing import Queue, Process

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    load_state_dict,
    safe_load,
    safe_save,
)
from tqdm import trange
import wandb
import cv2

from model import Model
from main import BASE_PATH


BS = 16
WARMUP_STEPS = 200
START_LR = 0.001
END_LR = 0.000001
STEPS = 500000


def loss_fn(pred, y):
    # return (pred[:, :3] - y[:, :3]).pow(2).sum()
    detected_loss = (pred[:, 0] - y[:, 0]).pow(2).mean()
    x_loss = (pred[:, 1] - y[:, 1]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    y_loss = (pred[:, 2] - y[:, 2]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    return detected_loss + x_loss + y_loss


@TinyJit
def train_step(x, y, lr):
    pred = model(x, y[:, 3].unsqueeze(1))
    loss = loss_fn(pred, y)

    optim.lr.assign(lr + 0.00001 - 0.00001).realize()
    optim.zero_grad()
    loss.backward()

    # calculate grad norm
    grad_norm = Tensor([0], requires_grad=False)
    for p in get_parameters(model):
        if p.grad is not None:
            grad_norm.assign(grad_norm + p.grad.pow(2).sum()).realize()

    optim.step()

    return loss.realize(), grad_norm.realize()


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
    while True:
        random.shuffle(preprocessed_train_files)
        for i in range(0, len(preprocessed_train_files) - BS, BS):
            batched = map(load_single_file, preprocessed_train_files[i : i + BS])
            x_b, y_b = zip(*batched)
            q.put((list(x_b), list(y_b)))


if __name__ == "__main__":
    Tensor.no_grad = False
    Tensor.training = True

    wandb.init(project="mrm_e2e_playground")
    wandb.config.update(
        {
            "warmup_steps": WARMUP_STEPS,
            "start_lr": START_LR,
            "end_lr": END_LR,
            "steps": STEPS,
        }
    )

    model = Model()

    sn_state_dict = safe_load("./weights/shufflenetv2.safetensors")
    load_state_dict(model.backbone, sn_state_dict)

    # state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
    # load_state_dict(model, state_dict)

    optim = AdamW(get_parameters(model), wd=0.0001)

    # start batch iterator in a separate process
    bi_queue = Queue(4)
    bi = Process(target=minibatch_iterator, args=(bi_queue,))
    bi.start()

    warming_up = True
    for step in (t := trange(STEPS)):
        if warming_up:
            new_lr = START_LR * (step / WARMUP_STEPS)
            if step >= WARMUP_STEPS:
                warming_up = False
        else:
            new_lr = END_LR + 0.5 * (START_LR - END_LR) * (
                1 + math.cos(((step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)) * math.pi)
            )

        if step == 0:
            x, y = bi_queue.get()
            x, y = Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)
        loss, grad_norm = train_step(x, y, Tensor([new_lr], requires_grad=False))
        x, y = bi_queue.get()
        x, y = Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)
        loss, grad_norm = loss.numpy().item(), grad_norm.numpy().item()
        t.set_description(f"loss: {loss:.6f}, lr: {optim.lr.numpy().item():.12f}, grad_norm: {grad_norm:.6f}")
        wandb.log(
            {
                "loss": loss,
                "lr": optim.lr.numpy().item(),
                "grad_norm": grad_norm,
            }
        )

        if step % 10000 == 0:
            safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{step}.safetensors"))
    safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))

    bi.terminate()
