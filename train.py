import glob
import math
import random
import gc

from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
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

from model import Model
from main import BASE_PATH


BS = 32
WARMUP_STEPS = 20
START_LR = 0.002
END_LR = 0.000001
STEPS = 2000 * 30


def loss_fn(pred, y):
    detected_loss = (pred[:, 0] - y[:, 0]).pow(2).sum()
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
    optim.step()
    return loss.realize()


preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.npz"))
preprocessed_train_data_loaded = [np.load(file) for file in preprocessed_train_files]


def minibatch_iterator():
    while True:
        random.shuffle(preprocessed_train_data_loaded)
        for chunk in preprocessed_train_data_loaded:
            # load chunk into memory
            chunk = {x: y for x, y in chunk.items()}
            order = list(range(0, chunk["y"].shape[0]))
            random.shuffle(order)
            for i in range(0, chunk["y"].shape[0] - BS, BS):
                yield (
                    Tensor(
                        chunk["x"][order[i : i + BS]],
                        requires_grad=False,
                        dtype=dtypes.float32,
                    ),
                    Tensor(
                        chunk["y"][order[i : i + BS]],
                        requires_grad=False,
                        dtype=dtypes.float32,
                    ),
                )


if __name__ == "__main__":
    from optimize import apply_optimizations_training

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
    apply_optimizations_training(model, BS)
    optim = LAMB(get_parameters(model), wd=0.0005)

    warming_up = True
    batch_iterator = minibatch_iterator()
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

        if step % 1000 == 0:
            safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))
    safe_save(get_state_dict(model), str(BASE_PATH / f"model.safetensors"))
