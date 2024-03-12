from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
import numpy as np
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import Context
from tinygrad import Device, Tensor, dtypes
from tinygrad import TinyJit
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad import GlobalCounters

from capture_and_display import ThreadedCapture, ThreadedOutput
from model import Model
from smoother import Smoother

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))
IMG_SIZE_W, IMG_SIZE_H = 256, 128

def resizeAndPad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  dtypes.default_float = dtypes.float16

  # cap_queue = Queue(4)
  # cap = ThreadedCapture(cap_queue, 1)
  # cap.start()

  # out_queue = Queue(4)
  # out = ThreadedOutput(out_queue)
  # out.start()

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  for key, param in get_state_dict(model).items():
    if "norm" in key: continue
    if "bn" in key: continue
    if "stage1.2" in key: continue
    if "stage5.2" in key: continue
    param.assign(param.half()).realize()
  smoother_x, smoother_y = Smoother(), Smoother()

  @TinyJit
  def pred(img):
    obj, pos = model(img)
    return obj[0, 0].float().realize(), pos[0, 0].float().realize()

  # cap = cv2.VideoCapture("2743.mp4")
  cap = cv2.VideoCapture(1)

  st = time.perf_counter()
  with Context(BEAM=4):
    while True:
      GlobalCounters.reset()
      # frame = cap_queue.get()

      ret, frame = cap.read()
      if not ret: break
      # convert to rgb
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # resize and pad
      frame = resizeAndPad(frame, (IMG_SIZE_H, IMG_SIZE_W))
      # crop center 256x128
      # frame = frame[256:256+IMG_SIZE_H, 256:256+IMG_SIZE_W]

      img = Tensor(frame).reshape(1, IMG_SIZE_H, IMG_SIZE_W, 3)
      obj, pos = pred(img)

      # show detection
      detected, x, y = obj.item(), pos[0].item(), pos[1].item()
      dt = time.perf_counter() - st
      st = time.perf_counter()
      cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      print(detected, x, y)
      x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
      cv2.putText(frame, f"{detected:.3f}, {x:.3f}, {y:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      if detected > 0.9:
        print(f"detected at {x}, {y}")
        x = x * IMG_SIZE_W
        y = y * IMG_SIZE_H
        cv2.circle(frame, (int(x), int(y)), 4, (0, 50, 255), -1)
        cv2.putText(frame, f"{int(x)}, {int(y)}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (55, 250, 55), 1)

      cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

      key = cv2.waitKey(1)
      if key == ord("q"): break

      time.sleep(0.05)
