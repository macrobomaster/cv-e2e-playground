from pathlib import Path
from multiprocessing import Queue
import time
import os

import cv2
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import safe_load, load_state_dict, get_parameters
from yolov8 import get_variant_multiples, Darknet
import onnxruntime as ort
import numpy as np

from capture_and_display import ThreadedCapture, ThreadedOutput
from model import Head
from utils import download_file
from smoother import Smoother
from optimize import apply_optimizations_inference


BASE_PATH = Path(os.environ.get("BASE_PATH", "./"))


def get_foundation():
    yolo_variant = "n"
    depth, width, ratio = get_variant_multiples(yolo_variant)
    net = Darknet(width, ratio, depth)

    weights_location = Path("./cache/") / f"yolov8{yolo_variant}.safetensors"
    download_file(
        f"https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors",
        weights_location,
    )

    state_dict = safe_load(str(weights_location))
    load_state_dict({"net": net}, state_dict)
    for param in get_parameters(net):
        param.assign(param.cast(dtypes.float32)).realize()

    def foundation(img):
        x = net(img.permute(0, 3, 1, 2).float() / 255)
        return x[-1]
    setattr(foundation, "net", net)

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

    # foundation = get_foundation()
    # head = Head()
    # load_state_dict(head, safe_load(str(BASE_PATH / "model.safetensors")))
    # apply_optimizations_inference(foundation, head)
    smoother_x, smoother_y = Smoother(), Smoother()

    # @TinyJit
    # def pred(img, color):
    #     return head(foundation(img), color)[0].realize()

    session = ort.InferenceSession("./model.onnx")

    cap = cv2.VideoCapture("2744.mp4")

    color = "red"
    st = time.perf_counter()
    while True:
        # frame = cap_queue.get()

        ret, frame = cap.read()
        if not ret:
            break
        # convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:, 106:746]

        # img = Tensor(frame).reshape(1, 480, 640, 3)
        # x = pred(img, Tensor([[0]]) if color == "red" else Tensor([[1]])).numpy()
        x = session.run(
            None,
            {
                "x": np.expand_dims(frame, 0).astype(np.float32),
                "color": np.array([[0]], dtype=np.int32)
                if color == "red"
                else np.array([[1]], dtype=np.int32),
            },
        )[0][0]

        # show detection
        detected, x, y, _ = x
        dt = time.perf_counter() - st
        st = time.perf_counter()
        cv2.putText(
            frame,
            f"{1/dt:.2f} FPS",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (55, 250, 55),
            2,
        )
        x, y = smoother_x.update(x, dt), smoother_y.update(y, dt)
        print(detected, x, y)
        if detected > 0.5:
            print(f"detected at {x}, {y}")
            # unscale to pixels
            x = x * 320 + 320
            y = y * 240 + 240
            cv2.circle(frame, (int(x), int(y)), 10, (0, 50, 255), -1)
            cv2.putText(
                frame,
                f"{int(x)}, {int(y)}",
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (55, 250, 55),
                2,
            )

        if color == "red":
            cv2.putText(
                frame,
                "detecting red",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (55, 250, 55),
                2,
            )
        elif color == "blue":
            cv2.putText(
                frame,
                "detecting blue",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (55, 250, 55),
                2,
            )
        cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("r"):
            color = "red"
        elif key == ord("b"):
            color = "blue"

        # time.sleep(0.025)
