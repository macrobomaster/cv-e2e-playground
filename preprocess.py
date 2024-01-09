import glob
from pathlib import Path
import random

import albumentations as A
import cv2
from tqdm import tqdm
import numpy as np

from main import BASE_PATH

IMG_SIZE = 320
DUPE_COUNT = 10
NON_DETECTED_RATIO = 0.15

PIPELINE = A.Compose([
  A.Perspective(p=0.25),
  A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
  A.RandomCrop(IMG_SIZE, IMG_SIZE),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
  A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
  A.CLAHE(p=0.1),
  A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
  A.RandomGamma(gamma_limit=(80, 120), p=0.1),
  A.FancyPCA(alpha=0.1, p=0.5),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

detected_count, non_detected_count = 0, 0
def add_to_batch(x_b, y_b, detected, color, img, x, y, raw=True):
  global detected_count, non_detected_count

  if color == -1: detected = 0

  if raw:
    if x < 0 or x >= IMG_SIZE or y < 0 or y >= IMG_SIZE: detected = 0
    # scale between 0 and 1
    x = x / IMG_SIZE
    y = y / IMG_SIZE

  if not detected:
    if random.random() > NON_DETECTED_RATIO: return
    x, y, color = 0.5, 0.5, random.randint(0, 1)
    non_detected_count += 1
  else: detected_count += 1

  x_b.append(np.reshape(img, (IMG_SIZE, IMG_SIZE, 3)))
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
      detected, x, y, color = (int(line[0]), int(line[1]), int(line[2]), int(line[3]))

    # add initial imgs
    x_b, y_b = [], []
    add_to_batch(x_b, y_b, detected, color, img[:IMG_SIZE, :IMG_SIZE], x, y)
    add_to_batch(x_b, y_b, detected, color, img[:IMG_SIZE, -IMG_SIZE:], IMG_SIZE - (img.shape[1] - x), y)
    add_to_batch(x_b, y_b, detected, color, img[-IMG_SIZE:, :IMG_SIZE], x, IMG_SIZE - (img.shape[0] - y))
    add_to_batch(x_b, y_b, detected, color, img[-IMG_SIZE:, -IMG_SIZE:], IMG_SIZE - (img.shape[1] - x), IMG_SIZE - (img.shape[0] - y))
    add_to_batch(x_b, y_b, detected, color, img[img.shape[0] // 2 - (IMG_SIZE // 2) : img.shape[0] // 2 + (IMG_SIZE // 2), img.shape[1] // 2 - (IMG_SIZE // 2) : img.shape[1] // 2 + (IMG_SIZE // 2)], IMG_SIZE // 2 - (img.shape[1] // 2 - x), IMG_SIZE // 2 - (img.shape[0] // 2 - y))

    # augment
    for _ in range(DUPE_COUNT):
      transformed = PIPELINE(image=img, keypoints=[(x, y)])
      add_to_batch(x_b, y_b, detected, color, transformed["image"], transformed["keypoints"][0][0], transformed["keypoints"][0][1])

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
    cv2.imwrite(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"), cv2.cvtColor(sx, cv2.COLOR_RGB2BGR))
    with open(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.txt"), "w") as f:
      f.write(" ".join(str(i) for i in sy))

print(f"detected: {detected_count}, non detected: {non_detected_count}")
print(f"Ratio: {detected_count / (detected_count + non_detected_count)}")
