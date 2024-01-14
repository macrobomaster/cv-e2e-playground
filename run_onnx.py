# import time
#
# import cv2
# import onnxruntime as ort
# import numpy as np
#
# from smoother import Smoother
#
# session_options = ort.SessionOptions()
# session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# session = ort.InferenceSession("model.onnx", session_options)
# smoother_x, smoother_y = Smoother(), Smoother()
#
# cap = cv2.VideoCapture("2744.mp4")
# # cap = cv2.VideoCapture(1)
#
# st = time.perf_counter()
# while True:
#   ret, frame = cap.read()
#   if not ret: break
#   # convert to rgb
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#   frame = frame[-320-50:-50, -320-200:-200]
#
#   x = session.run(None, {
#     "x": np.expand_dims(frame, 0).astype(np.float16),
#   })
#
#   # show detection
#   detected, x, y = x[0][0][0][0], x[1][0][0][0], x[1][0][0][1]
#   dt = time.perf_counter() - st
#   st = time.perf_counter()
#   cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 250, 55), 2)
#   x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
#   print(detected, x, y)
#   if detected > 0.9:
#       print(f"detected at {x}, {y}")
#       # unscale to pixels
#       x = x * 320
#       y = y * 320
#       cv2.circle(frame, (int(x), int(y)), 10, (0, 50, 255), -1)
#       cv2.putText(frame, f"{int(x)}, {int(y)}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 2)
#
#   cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#
#   key = cv2.waitKey(1)
#   if key == ord("q"): break

import time

import cv2
import onnxruntime as ort
import numpy as np

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = [("CUDAExecutionProvider", {"enable_cuda_graph": 1})]
session = ort.InferenceSession("model.onnx", sess_options, providers)

io_binding = session.io_binding()
x = ort.OrtValue.ortvalue_from_shape_and_type((1, 320, 320, 3), np.float16, "cuda", 0)
x_obj = ort.OrtValue.ortvalue_from_shape_and_type((1, 1, 1), np.float16, "cuda", 0)
x_pos = ort.OrtValue.ortvalue_from_shape_and_type((1, 1, 2), np.float16, "cuda", 0)
io_binding.bind_ortvalue_input("x", x)
io_binding.bind_ortvalue_output("x_obj", x_obj)
io_binding.bind_ortvalue_output("x_pos", x_pos)
session.run_with_iobinding(io_binding)

cap = cv2.VideoCapture("2744.mp4")
st = time.perf_counter()
frames = 0
while True:
  ret, frame = cap.read()
  if not ret: break
  # convert to rgb
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = frame[-320:, -320:]
  x.update_inplace(np.expand_dims(frame, 0).astype(np.float16))

  session.run_with_iobinding(io_binding)
  frames += 1
dt = time.perf_counter() - st
print("took", dt)
print("fps", frames/dt)
