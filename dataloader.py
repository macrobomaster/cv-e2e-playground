# Based off the tinygrad mlperf dataloader: https://github.com/tinygrad/tinygrad/blob/master/examples/mlperf/dataloader.py

import glob, random, signal, struct
from multiprocessing import Queue, Process, shared_memory, cpu_count

from PIL import Image
from tinygrad.helpers import Context, prod
from tinygrad import Tensor, dtypes
from tqdm import tqdm

from main import BASE_PATH

preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.png"))
def load_single_file(file):
  img = Image.open(file)
  img = img.convert("RGB") if img.mode != "RGB" else img

  # read the annotation file
  annotation_file = file.replace(".png", ".txt")
  with open(annotation_file, "r") as f:
    detected, x, y, color = f.readline().split(" ")
    detected, x, y, color = map(float, (detected, x, y, color))
  return img, (detected, x, y, color)

def shuffled_indices(n:int):
  indices = {}
  for i in range(n-1, -1, -1):
    j = random.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in:Queue, q_out:Queue, X:Tensor, Y:Tensor):
  signal.signal(signal.SIGINT, lambda *_: exit(0))
  with Context(DEBUG=0):
    while (recv := q_in.get()):
      idx, file = recv
      img, (detected, x, y, color) = load_single_file(file)

      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()
      Y[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = struct.pack("<ffff", detected, x, y, color)

      q_out.put(idx)
    q_out.put(None)

def batch_load(bs:int=32):
  BATCH_COUNT = len(preprocessed_train_files) // bs

  gen = shuffled_indices(len(preprocessed_train_files))
  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      file = preprocessed_train_files[next(gen)]
      q_in.put((idx, file))

  running = True
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if running:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while True:
      num = q_out.get()//bs
      gotten[num] += 1
      if gotten[num] == bs: break
    gotten[num] = 0
    return X[num*bs:(num+1)*bs], Y[num*bs:(num+1)*bs], Cookie(num)

  q_in, q_out = Queue(), Queue()
  X_sz = (BATCH_COUNT*bs, 128, 256, 3)
  X_shm = shared_memory.SharedMemory(name="e2e_train_X", create=True, size=prod(X_sz))
  Y_sz = (BATCH_COUNT*bs, 4)
  Y_shm = shared_memory.SharedMemory(name="e2e_train_Y", create=True, size=prod(Y_sz))
  procs = []
  try:
    X = Tensor.empty(*X_sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/{X_shm.name}")
    Y = Tensor.empty(*Y_sz, dtype=dtypes.float32, device=f"disk:/dev/shm/{Y_shm.name}")

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, Y))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    for _ in range(0, len(preprocessed_train_files)//bs): yield receive_batch()
  finally:
    running = False
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    for p in procs: p.terminate()
    X_shm.close()
    X_shm.unlink()
    Y_shm.close()
    Y_shm.unlink()

if __name__ == "__main__":
  BS = 256
  with tqdm(total=(len(preprocessed_train_files)//BS)*BS) as pbar:
    for x, y, _ in batch_load(BS):
      pbar.update(x.shape[0])
