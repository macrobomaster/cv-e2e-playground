import cv2
import sys
import queue
from multiprocessing import Process, Queue


class ThreadedCapture(Process):
    def __init__(self, q: Queue, src=0, frame_size=(640, 480)):
        super(ThreadedCapture, self).__init__()

        self.q = q

        self.cap = cv2.VideoCapture(src)
        assert self.cap.isOpened(), "Cannot open camera"
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 100)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 50)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_FPS, 120)

        self.killed = False

    def stop(self):
        self.killed = True

    def run(self):
        while not self.killed:
            g, f = self.cap.read()
            if g:
                try:
                    self.q.put_nowait(f)
                except queue.Full:
                    continue

        self.cap.release()


class ThreadedOutput(Process):
    def __init__(self, q: Queue):
        super(ThreadedOutput, self).__init__()

        self.q = q

        self.killed = False

    def stop(self):
        self.killed = True

    def run(self):
        while not self.killed:
            f = self.q.get()
            cv2.imshow("frame", f)
