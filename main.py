from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import safe_load, load_state_dict
from yolov8 import get_variant_multiples, Darknet, Yolov8NECK

from capture_and_display import ThreadedCapture, ThreadedOutput
from model import Head
from utils import download_file


BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))


def get_foundation():
    yolo_variant = "s"
    depth, width, ratio = get_variant_multiples(yolo_variant)
    net = Darknet(width, ratio, depth)
    fpn = Yolov8NECK(width, ratio, depth)

    weights_location = Path("/tmp") / f"yolov8{yolo_variant}.safetensors"
    download_file(
        f"https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors",
        weights_location,
    )

    state_dict = safe_load(str(weights_location))
    load_state_dict({"net": net, "fpn": fpn}, state_dict)

    def foundation(img):
        x = net(img.permute(0, 3, 1, 2).float() / 255)
        return fpn(*x)[-1]

    return foundation


if __name__ == "__main__":
    Tensor.no_grad = True
    Tensor.training = False

    # cap_queue = Queue(4)
    # cap = ThreadedCapture(cap_queue, 1)
    # cap.start()

    # out_queue = Queue(4)
    # out = ThreadedOutput(out_queue)
    # out.start()

    foundation = get_foundation()
    head = Head()
    load_state_dict(head, safe_load(str(BASE_PATH / "model.safetensors")))

    @TinyJit
    def pred(img):
        return head(foundation(img))[0].realize()

    cap = cv2.VideoCapture("2743.mp4")

    while True:
        # frame = cap_queue.get()

        ret, frame = cap.read()
        # convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:, 106:746]

        img = Tensor(frame).reshape(1, 480, 640, 3)
        x = pred(img)

        # show detection
        detected, x, y = x.numpy()
        print(detected, x, y)
        if detected > 0.5:
            print(f"detected at {x}, {y}")
            # unscale to pixels
            x = x * 320 + 320
            y = y * 240 + 240
            cv2.circle(frame, (int(x), int(y)), 10, (0, 50, 255), -1)

        cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        time.sleep(0.05)
