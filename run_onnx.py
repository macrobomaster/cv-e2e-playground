import time

import cv2
import onnxruntime as ort
import numpy as np

from smoother import Smoother

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.log_severity_level = 0
session = ort.InferenceSession("model.onnx", session_options)
smoother_x, smoother_y = Smoother(), Smoother()

cap = cv2.VideoCapture("2744.mp4")
# cap = cv2.VideoCapture(0)

st = time.perf_counter()
while True:
  ret, frame = cap.read()
  if not ret: break
  # convert to rgb
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = frame[-320-50:-50, -320-200:-200]

  x = session.run(None, {
    "x": np.expand_dims(frame, 0).astype(np.float16),
  })

  # show detection
  detected, x, y = x[0][0][0][0], x[1][0][0][0], x[1][0][0][1]
  dt = time.perf_counter() - st
  st = time.perf_counter()
  cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 250, 55), 2)
  x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
  print(detected, x, y)
  if detected > 0.9:
      print(f"detected at {x}, {y}")
      # unscale to pixels
      x = x * 320
      y = y * 320
      cv2.circle(frame, (int(x), int(y)), 10, (0, 50, 255), -1)
      cv2.putText(frame, f"{int(x)}, {int(y)}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 2)

  cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

  key = cv2.waitKey(1)
  if key == ord("q"): break
