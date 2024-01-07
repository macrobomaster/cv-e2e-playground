from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
from tinygrad.helpers import Context
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import GlobalCounters

from capture_and_display import ThreadedCapture, ThreadedOutput
from model import Model
from smoother import Smoother

BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  # cap_queue = Queue(4)
  # cap = ThreadedCapture(cap_queue, 1)
  # cap.start()

  # out_queue = Queue(4)
  # out = ThreadedOutput(out_queue)
  # out.start()

  model = Model()
  load_state_dict(model, safe_load(str(BASE_PATH / "model.safetensors")))
  smoother_x, smoother_y = Smoother(), Smoother()

  @TinyJit
  def pred(img):
    obj, pos = model(img)
    return obj[0, 0].realize(), pos[0, 0].realize()

  cap = cv2.VideoCapture("2744.mp4")

  st = time.perf_counter()
  with Context(BEAM=2):
    while True:
      GlobalCounters.reset()
      # frame = cap_queue.get()

      ret, frame = cap.read()
      if not ret: break
      # convert to rgb
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = frame[-256:, -256:]

      img = Tensor(frame).reshape(1, 256, 256, 3)
      obj, pos = pred(img)

      # show detection
      detected, x, y = obj.item(), pos[0].item(), pos[1].item()
      dt = time.perf_counter() - st
      st = time.perf_counter()
      cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
      print(detected, x, y)
      cv2.putText(frame, f"{detected:.3f}, {x:.3f}, {y:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)
      if True:
        print(f"detected at {x}, {y}")
        x = x * 256
        y = y * 256
        cv2.circle(frame, (int(x), int(y)), 10, (0, 50, 255), -1)
        cv2.putText(frame, f"{int(x)}, {int(y)}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 2)

      cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

      key = cv2.waitKey(1)
      if key == ord("q"): break

      time.sleep(0.05)
