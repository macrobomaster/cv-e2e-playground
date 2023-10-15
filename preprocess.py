import glob
from pathlib import Path
import random

import cv2
from tqdm import tqdm
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import getenv

from main import get_foundation, BASE_PATH


RANDOM_TRANSLATE_COUNT = 7
FLIP_COLOR_COUNT = 3


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


def add_to_batch(x_b, y_b, detected, color, img, x, y):
    if x < 0 or x >= 640 or y < 0 or y >= 480:
        detected = 0

    # scale between -1 and 1
    x = (x - 320) / 320
    y = (y - 240) / 240

    if not detected:
        x, y = 0, 0

    x_b.append(Tensor(img).reshape(480, 640, 3))
    y_b.append(Tensor([detected, x, y, color]))


train_files = glob.glob(str(BASE_PATH / "annotated/*.png"))
print(f"there are {len(train_files)} frames")


def get_train_data():
    for frame_file in train_files:
        # load x
        img = cv2.imread(frame_file)
        # convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_left = img[:, 212:852] if img.shape[1] > 640 else img
        img_right = img[:, 0:640]

        # load y
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
            line = f.readline().split(" ")
            detected, x, y, color = int(line[0]), int(line[1]), int(line[2]), int(line[3])

        # if the color is not known, make it random
        if color == -1:
            detected = 0
            color = random.randint(0, 1)

        # offset for crop
        x_left, x_right = x - 212 if img.shape[1] > 640 else x, x
        y_left, y_right = y, y

        # add initial imgs
        x_b, y_b = [], []
        add_to_batch(x_b, y_b, detected, color, img_left, x_left, y_left)
        add_to_batch(x_b, y_b, detected, color, img_right, x_right, y_right)

        # augment
        # random translate and pad
        for _ in range(RANDOM_TRANSLATE_COUNT):
            add_to_batch(
                x_b, y_b, detected, color, *random_translate(img_left, x_left, y_left)
            )
            add_to_batch(
                x_b, y_b, detected, color, *random_translate(img_right, x_right, y_right)
            )

        # flip color
        for _ in range(FLIP_COLOR_COUNT):
            add_to_batch(x_b, y_b, 0, 1 - color, *random_translate(img_left, x_left, y_left))
            add_to_batch(x_b, y_b, 0, 1 - color, *random_translate(img_right, x_right, y_right))

        # batch
        x = Tensor.stack(x_b, dim=0)
        y = Tensor.stack(y_b, dim=0)

        yield foundation_jit(x), y


# save into chunks
chunks = getenv("CHUNKS", 2)
chunk, chunk_x, chunk_y = 0, [], []
for i, (x, y) in enumerate(tqdm(get_train_data(), total=len(train_files))):
    chunk_x.append(x.numpy())
    chunk_y.append(y.numpy())
    if (i + 1) % (len(train_files) // chunks) == 0:
        print(f"saving chunk {chunk}")
        np.savez(str(BASE_PATH / f"preprocessed/{chunk}.npz"), x=np.concatenate(chunk_x).astype(np.float16), y=np.concatenate(chunk_y).astype(np.float16))
        chunk_x, chunk_y = [], []
        chunk += 1
