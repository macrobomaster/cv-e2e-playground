import time

import cv2
import onnxruntime as ort
import numpy as np

from smoother import Smoother


if __name__ == "__main__":
    session = ort.InferenceSession("model_fp16.onnx")
    smoother_x, smoother_y = Smoother(), Smoother()

    cap = cv2.VideoCapture("2744.mp4")

    color = "red"
    st = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:, 106:746]
        # frame = frame[:, 212:]

        x = session.run(
            None,
            {
                "x": np.expand_dims(frame, 0).astype(np.float32),
                "color": np.array([[0]], dtype=np.int32)
                if color == "red"
                else np.array([[1]], dtype=np.int32),
            },
        )[0]

        # show detection
        detected, x, y, _ = x[0]
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
