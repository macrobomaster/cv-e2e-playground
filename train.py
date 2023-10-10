import glob
import math
import random

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


WARMUP_STEPS = 20
START_LR = 0.002
END_LR = 0.00001
EPOCHS = 200
STEPS = 200


def loss_fn(pred, y):
    detected_loss = (pred[:, 0] - y[:, 0]).pow(2).sum()
    x_loss = (pred[:, 1] - y[:, 1]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    y_loss = (pred[:, 2] - y[:, 2]).mul(pred[:, 0] + y[:, 0] + 0.5).pow(2).sum()
    return detected_loss + x_loss + y_loss


@TinyJit
def train_step(x, y, lr):
    pred = head(x)
    loss = loss_fn(pred, y)
    optim.lr.assign(lr + 0.00001 - 0.00001).realize()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.realize()


preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.npz"))
preprocessed_train_data = [np.load(file) for file in preprocessed_train_files]
preprocessed_batch_size = preprocessed_train_data[0]["y"].shape[0]


def get_minibatch(size=4):
    x_b, y_b = [], []
    for _ in range(size):
        data = random.choice(preprocessed_train_data)
        sel = random.randint(0, preprocessed_batch_size - 1)  # type: ignore
        try:
            x_b.append(data["x"][sel])
            y_b.append(data["y"][sel])
        except:
            print(sel)
            exit(1)
    return Tensor(np.stack(x_b)), Tensor(np.stack(y_b))


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
    optim = LAMB(get_parameters(head))

    warming_up = True
    for epoch in (t := trange(EPOCHS)):
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
                x, y = get_minibatch()
            loss = train_step(x, y, Tensor([new_lr]))
            if step != 0:
                x, y = get_minibatch()
            loss = loss.numpy().item()
            t.set_description(f"loss: {loss:.6f}, lr: {optim.lr.numpy().item():.12f}")
            wandb.log(
                {
                    "loss": loss,
                    "lr": optim.lr.numpy().item(),
                }
            )

        safe_save(get_state_dict(head), str(BASE_PATH / f"model.safetensors"))
