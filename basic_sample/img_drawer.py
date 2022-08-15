from random import random, randint
import cv2
import numpy as np

# draw mosaic
# img = np.empty((300,300,3), np.uint8)
# for row in range(300):
#     for col in range(300):
#         img[row][col] = [randint(0, 255), randint(0, 255), randint(0, 255)]
# cv2.imshow("draw", img)
# cv2.waitKey(5000)

# cut img
# img = cv2.imread("./img/13.md-a1e8396741da620ab3c6ed05391e85cf.jpg")
# cut_img = img[:200, :400]
# cv2.imshow("cut", cut_img)
# cv2.imshow("draw", img)
# cv2.waitKey(0)

# write Geometry & text
img = np.zeros((600, 600, 3), np.uint8)
cv2.line(img, (0, 0), (500, 550), color=(255, 0, 0), thickness=3)
cv2.circle(img, (300, 300), radius=50, color=(0, 255, 0), thickness=cv2.FILLED)
cv2.rectangle(img, (100, 100), (300, 300), color=(0, 0, 255), thickness=2)
cv2.putText(
    img=img, 
    text="My First Text", 
    org= (100, 400),
    fontFace=cv2.FONT_HERSHEY_DUPLEX,
    fontScale=1,
    color=(255,255,255)
)
cv2.imshow("drawler", img)
cv2.waitKey(0)
