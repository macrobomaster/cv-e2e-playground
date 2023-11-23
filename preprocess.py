import glob
from pathlib import Path
import random

import cv2
from tqdm import tqdm
import numpy as np

from main import BASE_PATH


RANDOM_TRANSLATE_COUNT = 0
FLIP_COLOR_COUNT = 1


def random_translate(img, x, y):
    img = img.copy()

    # random translate within half the image
    rand_x = random.randint(-img.shape[1] // 2, img.shape[1] // 2)
    rand_y = random.randint(-img.shape[0] // 2, img.shape[0] // 2)

    # pad the image because the translate might go out of bounds
    img = cv2.copyMakeBorder(img, 176, 176, 176, 176, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # crop a 352x352 square around the new center
    img = img[
        img.shape[0] // 2 + rand_y - 176 : img.shape[0] // 2 + rand_y + 176,
        img.shape[1] // 2 + rand_x - 176 : img.shape[1] // 2 + rand_x + 176,
    ]

    return img, x, y


detected_count, non_detected_count = 0, 0


def add_to_batch(x_b, y_b, detected, color, img, x, y):
    global detected_count, non_detected_count

    if color == -1:
        detected = 0

    if x < 0 or x >= 352 or y < 0 or y >= 352:
        detected = 0

    # scale between -1 and 1
    x = (x - 176) / 176
    y = (y - 176) / 176

    if not detected:
        x, y, color = 0, 0, random.randint(0, 1)
        non_detected_count += 1
    else:
        detected_count += 1

    x_b.append(np.reshape(img, (352, 352, 3)))
    y_b.append((detected, x, y, color))


train_files = glob.glob(str(BASE_PATH / "annotated/*.png"))
print(f"there are {len(train_files)} frames")


def get_train_data(only_detected=False):
    global non_detected_count
    for frame_file in train_files:
        # load x
        img = cv2.imread(frame_file)
        # convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load y
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
            line = f.readline().split(" ")
            detected, x, y, color = (
                int(line[0]),
                int(line[1]),
                int(line[2]),
                int(line[3]),
            )

        # add initial imgs
        x_b, y_b = [], []
        add_to_batch(x_b, y_b, detected, color, img[:352, :352], x, y)
        add_to_batch(x_b, y_b, detected, color, img[:352, -352:], 352 - (img.shape[1] - x), y)
        add_to_batch(x_b, y_b, detected, color, img[-352:, :352], x, 352 - (img.shape[0] - y))
        add_to_batch(x_b, y_b, detected, color, img[-352:, -352:], 352 - (img.shape[1] - x), 352 - (img.shape[0] - y))
        add_to_batch(x_b, y_b, detected, color, img[img.shape[0] // 2 - 176 : img.shape[0] // 2 + 176, img.shape[1] // 2 - 176 : img.shape[1] // 2 + 176], 352 // 2 - (img.shape[1] // 2 - x), 352 // 2 - (img.shape[0] // 2 - y))

        # augment
        # random translate and pad
        for _ in range(RANDOM_TRANSLATE_COUNT):
            add_to_batch(x_b, y_b, detected, color, *random_translate(img, x, y))

        # flip color
        for _ in range(FLIP_COLOR_COUNT):
            # add_to_batch(x_b, y_b, 0, 1 - color, *random_translate(img, x, y))
            add_to_batch(x_b, y_b, 0, 1 - color, img[:352, :352], x, y)
            add_to_batch(x_b, y_b, 0, 1 - color, img[:352, -352:], 352 - (img.shape[1] - x), y)
            add_to_batch(x_b, y_b, 0, 1 - color, img[-352:, :352], x, 352 - (img.shape[0] - y))
            add_to_batch(x_b, y_b, 0, 1 - color, img[-352:, -352:], 352 - (img.shape[1] - x), 352 - (img.shape[0] - y))
            add_to_batch(x_b, y_b, 0, 1 - color, img[img.shape[0] // 2 - 176 : img.shape[0] // 2 + 176, img.shape[1] // 2 - 176 : img.shape[1] // 2 + 176], 352 // 2 - (img.shape[1] // 2 - x), 352 // 2 - (img.shape[0] // 2 - y))

        # adjust brightness and contrast
        for i in range(len(x_b)):
            x_b[i] = cv2.convertScaleAbs(x_b[i], alpha=random.uniform(0.8, 1.2), beta=random.uniform(-50, 50))

        # adjust saturation
        for i in range(len(x_b)):
            hsv = cv2.cvtColor(x_b[i], cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            x_b[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # filter out non detected if only detected
        if only_detected:
            try:
                nx_b, ny_b = zip(*[(x, y) for x, y in zip(x_b, y_b) if y[0] == 1])
                # fix the non detected count
                difference = len(x_b) - len(nx_b)
                non_detected_count -= difference
                x_b, y_b = nx_b, ny_b
            except ValueError:
                non_detected_count -= len(x_b)
                x_b, y_b = [], []

        yield x_b, y_b


for i, (x, y) in enumerate(tqdm(get_train_data(False), total=len(train_files))):
    for j, (sx, sy) in enumerate(zip(x, y)):
        cv2.imwrite(
            str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"),
            cv2.cvtColor(sx, cv2.COLOR_RGB2BGR),
        )
        with open(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.txt"), "w") as f:
            f.write(" ".join(str(i) for i in sy))

for dupe in range(3):
    for i, (x, y) in enumerate(tqdm(get_train_data(True), total=len(train_files))):
        i += len(train_files) * (dupe + 1)
        for j, (sx, sy) in enumerate(zip(x, y)):
            cv2.imwrite(
                str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"),
                cv2.cvtColor(sx, cv2.COLOR_RGB2BGR),
            )
            with open(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.txt"), "w") as f:
                f.write(" ".join(str(i) for i in sy))

print(f"detected: {detected_count}, non detected: {non_detected_count}")
print(f"Ratio: {detected_count / (detected_count + non_detected_count)}")
