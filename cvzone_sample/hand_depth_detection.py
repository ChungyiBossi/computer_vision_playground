import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Manually meansure
# Find Function
image_dot_distance = [300, 245, 200, 170, 145, 130,
                      112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
real_distance_in_CM = [20, 25, 30, 35, 40, 45,
                       50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(image_dot_distance, real_distance_in_CM,
                  2)  # y = Ax^2 + Bx + C

# Loop
while True:
    success, img = cap.read()
    # hands = detector.findHands(img, draw=False)
    hands, drawed_img = detector.findHands(img, draw=True)
    for hand in hands:
        lmList = hand['lmList']
        x, y, w, h = hand['bbox']
        x1, y1 = lmList[5][:2]
        x2, y2 = lmList[17][:2]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        # print(distanceCM, distance)

        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x, y+h+10))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
