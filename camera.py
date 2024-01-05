import cv2
from capture_and_display import ThreadedCapture

from multiprocessing import Queue
import time

cap_queue = Queue(4)
cap = ThreadedCapture(cap_queue, 5)
cap.start()

st = time.perf_counter()
writer = None
record = False
while True:
  frame = cap_queue.get()

  dt = time.perf_counter() - st
  st = time.perf_counter()

  if record and writer is not None:
    writer.write(frame)

  cv2.putText(frame, f"{1/dt:.2f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 250, 55), 2)
  cv2.putText(frame, f"{'RECORDING' if record else ''}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 250, 55), 2)

  cv2.imshow("frame", frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("r"):
    # start recording
    writer = cv2.VideoWriter(f"./record/{time.time()}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    record = True
  elif key == ord("s") and writer is not None:
    # stop recording
    record = False
    writer.release()

cap.kill()
cv2.destroyAllWindows()
