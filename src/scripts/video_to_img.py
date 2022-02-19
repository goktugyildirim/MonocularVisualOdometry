import cv2
import numpy as np


vidcap = cv2.VideoCapture('/home/goktug/projects/MonocularVO/src/images_test/videos/t1.mp4')
success,image = vidcap.read()
count = 0
while success:
  sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  sharpen = cv2.filter2D(image, -1, sharpen_kernel)
  cv2.imwrite("%d.jpg" % count, sharpen)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
