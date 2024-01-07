import glob
from pathlib import Path
import random

import cv2
from tqdm import tqdm
import numpy as np

from main import BASE_PATH

IMG_SIZE = 256
RANDOM_TRANSLATE_COUNT = 0
FLIP_COLOR_COUNT = 0

def random_translate(img, x, y):
  img = img.copy()

  # random translate within half the image
  rand_x = random.randint(-img.shape[1] // 2, img.shape[1] // 2)
  rand_y = random.randint(-img.shape[0] // 2, img.shape[0] // 2)

  # pad the image because the translate might go out of bounds
  img = cv2.copyMakeBorder(img, IMG_SIZE // 2, IMG_SIZE // 2, IMG_SIZE // 2, IMG_SIZE // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

  # crop a IMG_SIZExIMG_SIZE square around the new center
  img = img[
    img.shape[0] // 2 + rand_y - IMG_SIZE // 2 : img.shape[0] // 2 + rand_y + IMG_SIZE // 2,
    img.shape[1] // 2 + rand_x - IMG_SIZE // 2 : img.shape[1] // 2 + rand_x + IMG_SIZE // 2,
  ]

  return img, x, y

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
    if random.random() > 0.2: return
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
    # random translate and pad
    for _ in range(RANDOM_TRANSLATE_COUNT): add_to_batch(x_b, y_b, detected, color, *random_translate(img, x, y))

    # flip color
    for _ in range(FLIP_COLOR_COUNT): add_to_batch(x_b, y_b, 0, 1 - color, *random_translate(img, x, y))

    # brightness and contrast
    for i in range(len(x_b)):
      add_to_batch(x_b, y_b, y_b[i][0], y_b[i][3], cv2.convertScaleAbs(x_b[i], alpha=random.uniform(0.8, 1.2), beta=random.uniform(-50, 50)), y_b[i][1], y_b[i][2], raw=False)

    # adjust saturation
    for i in range(len(x_b)):
      hsv = cv2.cvtColor(x_b[i], cv2.COLOR_RGB2HSV)
      hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
      hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
      add_to_batch(x_b, y_b, y_b[i][0], y_b[i][3], cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), y_b[i][1], y_b[i][2], raw=False)

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

# for dupe in range(3):
#   for i, (x, y) in enumerate(tqdm(get_train_data(True), total=len(train_files))):
#     i += len(train_files) * (dupe + 1)
#     for j, (sx, sy) in enumerate(zip(x, y)):
#       cv2.imwrite(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.png"), cv2.cvtColor(sx, cv2.COLOR_RGB2BGR))
#       with open(str(BASE_PATH / f"preprocessed/{i:06}_{j:03}.txt"), "w") as f:
#         f.write(" ".join(str(i) for i in sy))

print(f"detected: {detected_count}, non detected: {non_detected_count}")
print(f"Ratio: {detected_count / (detected_count + non_detected_count)}")
