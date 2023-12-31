import os

import cv2
import glob
from pathlib import Path

from main import BASE_PATH

if (stage := int(os.environ.get("STAGE", 0))) == 0:
  frame_files = glob.glob(str(BASE_PATH / "data/*.png"))
  print(f"there are {len(frame_files)} frames")

  oframe, frame = None, None
  click_pos = None

  def click_handler(event, x, y, flags, param):
    global oframe, frame, click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
      frame = oframe.copy()  # type: ignore
      cv2.circle(frame, (x, y), 3, (100, 0, 255), -1)
      click_pos = (x, y)

  cv2.namedWindow("preview")
  cv2.setMouseCallback("preview", click_handler)

  i = 0
  flag = False
  while not flag and i < len(frame_files):
    print(f"annotating frame {i} of {len(frame_files)}")
    frame_file = frame_files[i]
    oframe = frame = cv2.imread(frame_file)

    while True:
      cv2.imshow("preview", frame)

      key = cv2.waitKey(1)
      if key == ord("q"):
        flag = True
        break
      elif key == ord("s"):
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"0 0 0")
        break
      elif key == ord("a") and click_pos is not None:
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"1 {click_pos[0]} {click_pos[1]}")  # type: ignore
        click_pos = None
        break
      elif key == ord("d"):
        i -= 2
        break
    i += 1
elif stage == 1:
  frame_files = glob.glob(str(BASE_PATH / "data/*.png"))
  print(f"there are {len(frame_files)} frames")

  i = 0
  flag = False
  while not flag and i < len(frame_files):
    print(f"annotating color of frame {i} of {len(frame_files)}")
    frame_file = frame_files[i]
    frame = cv2.imread(frame_file)
    cv2.imshow("preview", frame)

    while True:
      key = cv2.waitKey(0)
      if key == ord("q"):
        flag = True
        break
      elif key == ord("s"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y = int(line[0]), int(line[1]), int(line[2])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} -1")
        break
      elif key == ord("r"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y = int(line[0]), int(line[1]), int(line[2])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} 0")
        break
      elif key == ord("b"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y = int(line[0]), int(line[1]), int(line[2])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} 1")
        break
      elif key == ord("d"):
        i -= 2
        break
    i += 1
elif stage == 2:
  frame_files = glob.glob(str(BASE_PATH / "data/*.png"))
  print(f"there are {len(frame_files)} frames")

  i = 0
  flag = False
  while not flag and i < len(frame_files):
    print(f"annotating number of frame {i} of {len(frame_files)}")
    frame_file = frame_files[i]
    frame = cv2.imread(frame_file)
    cv2.imshow("preview", frame)

    while True:
      key = cv2.waitKey(0)
      if key == ord("q"):
        flag = True
        break
      elif key == ord("s"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y, color = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} {color} -1")
        break
      elif key == ord("1"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y, color = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} {color} 1")
        break
      elif key == ord("3"):
        with open(Path(frame_file).with_suffix(".txt"), "r") as f:
          line = f.readline().split(" ")
          detected, x, y, color = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"{detected} {x} {y} {color} 3")
        break
      elif key == ord("d"):
        i -= 2
        break
    i += 1
