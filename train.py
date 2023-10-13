import glob
import math
import random
import gc

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import LAMB
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    load_state_dict,
    safe_load,
    safe_save,
)
from tqdm import trange
import numpy as np
import wandb

from model import Head
from main import BASE_PATH
from optimize import apply_optimizations_training


BS = 32
WARMUP_STEPS = 20
START_LR = 0.002
END_LR = 0.000001
EPOCHS = 20
STEPS = 1000


def loss_fn(pred, y):
    detected_loss = (pred[:, 0] - y[:, 0]).pow(2).sum()
    x_loss = (pred[:, 1] - y[:, 1]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    y_loss = (pred[:, 2] - y[:, 2]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    return detected_loss + x_loss + y_loss


@TinyJit
def train_step(x, y, lr):
    pred = head(x, y[:, 3].unsqueeze(1))
    loss = loss_fn(pred, y)
    optim.lr.assign(lr + 0.00001 - 0.00001).realize()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.realize()


preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.npz"))
preprocessed_train_data_loaded = [np.load(file) for file in preprocessed_train_files]
preprocessed_train_data = [
    {x: y for x, y in data.items()} for data in preprocessed_train_data_loaded
]
for data in preprocessed_train_data_loaded:
    data.close()


def minibatch_iterator():
    while True:
        random.shuffle(preprocessed_train_data)
        for chunk in preprocessed_train_data:
            order = list(range(0, chunk["y"].shape[0]))
            random.shuffle(order)
            for i in range(0, chunk["y"].shape[0] - BS, BS):
                yield Tensor(
                    chunk["x"][order[i : i + BS]], requires_grad=False
                ), Tensor(chunk["y"][order[i : i + BS]], requires_grad=False)


if __name__ == "__main__":
    Tensor.no_grad = False
    Tensor.training = True

    wandb.init(project="mrm_e2e_playground")
    wandb.config.update(
        {
            "warmup_steps": WARMUP_STEPS,
            "start_lr": START_LR,
            "end_lr": END_LR,
            "epochs": EPOCHS,
            "steps": STEPS,
        }
    )

    head = Head()
    # load_state_dict(head, safe_load("model.safetensors"))
    apply_optimizations_training(head, BS)
    optim = LAMB(get_parameters(head), wd=0.0001)

    warming_up = True
    for epoch in (t := trange(EPOCHS)):
        batch_iterator = minibatch_iterator()
        for step in range(STEPS):
            if warming_up:
                new_lr = START_LR * (step / WARMUP_STEPS)
                if step >= WARMUP_STEPS:
                    warming_up = False
            else:
                new_lr = END_LR + 0.5 * (START_LR - END_LR) * (
                    1
                    + math.cos(
                        (
                            (step + ((epoch * STEPS) - WARMUP_STEPS))
                            / ((EPOCHS * STEPS) - WARMUP_STEPS)
                        )
                        * math.pi
                    )
                )

            if step == 0:
                x, y = next(batch_iterator)
            loss = train_step(x, y, Tensor([new_lr], requires_grad=False))
            x, y = next(batch_iterator)
            loss = loss.numpy().item()
            t.set_description(f"loss: {loss:.6f}, lr: {optim.lr.numpy().item():.12f}")
            wandb.log(
                {
                    "loss": loss,
                    "lr": optim.lr.numpy().item(),
                }
            )

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            safe_save(get_state_dict(head), str(BASE_PATH / f"model.safetensors"))
