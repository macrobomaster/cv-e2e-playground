import glob
from pathlib import Path
import random

import albumentations as A
import cv2
from tqdm import tqdm
import numpy as np

from main import BASE_PATH

IMG_SIZE_W = 256
IMG_SIZE_H = 128
DUPE_COUNT = 10
NON_DETECTED_RATIO = 0.6
TOTAL_COUNT = 10000

PIPELINE = A.Compose([
  A.Perspective(p=0.25),
  A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
  A.OneOf([
    A.RandomCrop(IMG_SIZE_H, IMG_SIZE_W, p=0.2),
    A.Compose([
      A.LongestMaxSize(max_size=IMG_SIZE_W, p=1),
      A.RandomCrop(IMG_SIZE_H, IMG_SIZE_W, p=1)
    ], p=0.8),
  ], p=1),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
  A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
  A.CLAHE(p=0.1),
  A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
  A.RandomGamma(gamma_limit=(80, 120), p=0.1),
  A.FancyPCA(alpha=0.1, p=0.5),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

detected_count, non_detected_count = 0, 0
def add_to_batch(x_b, y_b, detected, color, img, x, y, raw=True, add_non_detected=True):
  global detected_count, non_detected_count

  if color == -1: detected = 0

  if raw:
    if x < 0 or x >= IMG_SIZE_W or y < 0 or y >= IMG_SIZE_H: detected = 0
    # scale between 0 and 1
    x = x / IMG_SIZE_W
    y = y / IMG_SIZE_H

  if not detected:
    x, y, color = 0.5, 0.5, random.randint(0, 1)
    if add_non_detected: non_detected_count += 1
  else: detected_count += 1

  if not add_non_detected and detected == 0: return False
  x_b.append(np.reshape(img, (IMG_SIZE_H, IMG_SIZE_W, 3)))
  y_b.append((detected, x, y, color))
  return True

def get_raw_data(frame_file):
  # load x
  img = cv2.imread(frame_file)
  # convert to rgb
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # load y
  with open(Path(frame_file).with_suffix(".txt"), "r") as f:
    line = f.readline().split(" ")
    detected, x, y, color = (int(line[0]), int(line[1]), int(line[2]), int(line[3]))
  return img, detected, x, y, color

train_files = glob.glob(str(BASE_PATH / "annotated/*.png"))
print(f"there are {len(train_files)} frames")
def get_train_data():
  global non_detected_count
  file_index = 0
  added_count = 0
  while added_count < TOTAL_COUNT:
    x_b, y_b = [], []

    # first add only detected
    for _ in range(int(DUPE_COUNT * NON_DETECTED_RATIO)):
      added = False
      while not added:
        img, detected, x, y, color = get_raw_data(train_files[file_index])
        transformed = PIPELINE(image=img, keypoints=[(x, y)])
        added = add_to_batch(x_b, y_b, detected, color, transformed["image"], transformed["keypoints"][0][0], transformed["keypoints"][0][1], add_non_detected=False)
        file_index = (file_index + 1) % len(train_files)

    # then add the rest
    for _ in range(int(DUPE_COUNT * (1 - NON_DETECTED_RATIO))):
      img, detected, x, y, color = get_raw_data(train_files[file_index])
      transformed = PIPELINE(image=img, keypoints=[(x, y)])
      add_to_batch(x_b, y_b, detected, color, transformed["image"], transformed["keypoints"][0][0], transformed["keypoints"][0][1])
      file_index = (file_index + 1) % len(train_files)

    added_count += DUPE_COUNT
    yield x_b, y_b

for i, (x, y) in enumerate(tqdm(get_train_data(), total=TOTAL_COUNT // DUPE_COUNT)):
  for j, (sx, sy) in enumerate(zip(x, y)):
    cv2.imwrite(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"), cv2.cvtColor(sx, cv2.COLOR_RGB2BGR))
    with open(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.txt"), "w") as f:
      f.write(" ".join(str(i) for i in sy))

print(f"detected: {detected_count}, non detected: {non_detected_count}")
print(f"Ratio: {detected_count / (detected_count + non_detected_count)}")
