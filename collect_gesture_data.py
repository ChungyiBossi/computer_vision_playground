import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
# Costumize Gesture
gesture_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "x"]
folder = "./gesture_data"
counter = 0
key_press = ""
while True:
    success, img = cap.read()
    if success:
        hands, img = detector.findHands(img)
        if hands:
            # 下方為將手勢部位特別擷取出來，並fix成”不變形狀“的大小
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(y - offset, 0):y + h + offset,
                          max(x - offset, 0):x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
            except Exception as e:
                print("Exception Occur:", e)
                print("hand bbox (x, y, w, h): ", (x, y, w, h))
                print("img shape: ", img.shape)
                print("imgCrop shape: ", imgCropShape)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        if key_press:
            cv2.putText(img, f"Key Press: {key_press}", (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(10)
        for gs in gesture_string:
            if key == ord(gs):
                key_press = gs
                if not os.path.exists(f"{folder}/{gs}"):
                    print("Create corresponding data folder......:")
                    os.mkdir(f"{folder}/{gs}")
                counter += 1
                cv2.imwrite(f'{folder}/{gs}/Image_{time.time()}.jpg', imgWhite)
                cv2.waitKey(300)
        if key not in [ord(gs) for gs in gesture_string]:
            key_press = ""

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
