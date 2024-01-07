import cv2
import glob

from main import BASE_PATH

preprocessed_train_files = glob.glob(str(BASE_PATH / "preprocessed/*.png"))

i = 0
while i < len(preprocessed_train_files):
  file = preprocessed_train_files[i]
  img = cv2.imread(file)
  # read the annotation file
  annotation_file = file.replace(".png", ".txt")
  with open(annotation_file, "r") as f:
    detected, x, y, color = f.readline().split(" ")
    detected, color = int(detected), int(color)
    x, y = int(float(x) * img.shape[1]), int(float(y) * img.shape[0])

  # draw the annotation
  if detected: cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
  cv2.putText(img, "color: " + str(color), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  cv2.putText(img, "detected: " + str(detected), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  cv2.imshow("img", img)

  key = cv2.waitKey(0)
  if key == ord("q"): break
  elif key == ord("a"): i -= 1
  else: i += 1

cv2.destroyAllWindows()
