import cv2
from matplotlib import pyplot as plt
import numpy as np
image = cv2.imread("zy2\image_2.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  
  
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

cnt_len = cv2.arcLength(contours[0], True)
cnt = cv2.approxPolyDP(contours[0], 0.02*cnt_len, True)
if len(cnt) == 4:
    cv2.drawContours(image, [cnt], -1, (255, 255, 0), 3 )
plt.imshow(image)
plt.show()
