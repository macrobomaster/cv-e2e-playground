import glob
from pathlib import Path
import random

import cv2
from tqdm import tqdm
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit

from main import get_foundation
from train import BASE_PATH


RANDOM_TRANSLATE_COUNT = 7


foundation = get_foundation()
foundation_jit = TinyJit(lambda x: foundation(x).realize())


def random_translate(img, x, y):
    img = img.copy()
    rand_x = random.randint(-320, 320)
    rand_y = random.randint(-240, 240)

    # crop image
    if rand_x > 0:
        img = img[:, :-rand_x]
    elif rand_x < 0:
        img = img[:, -rand_x:]
    if rand_y > 0:
        img = img[:-rand_y, :]
    elif rand_y < 0:
        img = img[-rand_y:, :]
    img = cv2.copyMakeBorder(
        img,
        rand_y if rand_y > 0 else 0,
        -rand_y if rand_y < 0 else 0,
        rand_x if rand_x > 0 else 0,
        -rand_x if rand_x < 0 else 0,
        cv2.BORDER_REPLICATE,
    )

    # adjust x and y
    x = x + rand_x
    y = y + rand_y

    return img, x, y


def add_to_batch(x_b, y_b, detected, img, x, y):
    if x < 0 or x >= 640 or y < 0 or y >= 480:
        detected = 0

    # scale between -1 and 1
    x = (x - 320) / 320
    y = (y - 240) / 240

    if not detected:
        x, y = 0, 0

    x_b.append(Tensor(img).reshape(480, 640, 3))
    y_b.append(Tensor([detected, x, y]))


train_files = glob.glob(str(BASE_PATH / "annotated/*.png"))
print(f"there are {len(train_files)} frames")


def get_train_data():
    for frame_file in train_files:
        # load x
        img = cv2.imread(frame_file)
        # convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_left = img[:, 212:852]
        img_right = img[:, 0:640]

        # load y
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
            line = f.readline()
            line = line.split(" ")
            detected, x, y = int(line[0]), int(line[1]), int(line[2])

        # offset for crop
        x_left, x_right = x - 212, x
        y_left, y_right = y, y

        # add initial imgs
        x_b, y_b = [], []
        add_to_batch(x_b, y_b, detected, img_left, x_left, y_left)
        add_to_batch(x_b, y_b, detected, img_right, x_right, y_right)

        # augment
        # random translate and pad
        for _ in range(RANDOM_TRANSLATE_COUNT):
            add_to_batch(
                x_b, y_b, detected, *random_translate(img_left, x_left, y_left)
            )
            add_to_batch(
                x_b, y_b, detected, *random_translate(img_right, x_right, y_right)
            )

        # final step for augment is to generate flips
        for i in range(len(x_b)):
            x_b.append(x_b[i][:, ::-1])
            detected, x, y = y_b[i].numpy()
            y_b.append(Tensor([detected, -x, y]))

        # batch
        x = Tensor.stack(x_b, dim=0)
        y = Tensor.stack(y_b, dim=0)

        yield foundation_jit(x), y


for i, (x, y) in enumerate(tqdm(get_train_data(), total=len(train_files))):
    np.savez(str(BASE_PATH / f"preprocessed/{i}.npz"), x=x.numpy(), y=y.numpy())
