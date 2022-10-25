import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


# Costumize Gesture
gesture_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "x"]
folder = "./gesture_data"


def capture_hand_part(img, hand_data, img_size=300, cap_offset=20):
    x, y, w, h = hand_data['bbox']

    imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
    img_hand_part = img[max(y - cap_offset, 0):y + h + cap_offset,
                        max(x - cap_offset, 0):x + w + cap_offset]
    img_hand_partShape = img_hand_part.shape
    aspectRatio = h / w

    try:
        if aspectRatio > 1:
            k = img_size / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(img_hand_part, (wCal, img_size))
            wGap = math.ceil((img_size - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = img_size / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(img_hand_part, (img_size, hCal))
            hGap = math.ceil((img_size - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
    except Exception as e:
        print("Exception Occur:", e)
        print("hand bbox (x, y, w, h): ", (x, y, w, h))
        print("img shape: ", img.shape)
        print("img_hand_part shape: ", img_hand_partShape)

    return imgWhite


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    counter = 0
    key_press = ""
    while True:
        success, img = cap.read()
        if success:
            hands, img = detector.findHands(img)
            if hands:
                # 下方為將手勢部位特別擷取出來，並fix成”不變形狀“的大小
                hand_data = hands[0]
                x = hand_data['bbox'][0]
                y = hand_data['bbox'][1]
                imgWhite = capture_hand_part(img, hand_data)
                cv2.imshow("ImageWhite", imgWhite)

            if key_press:
                cv2.putText(img, f"Key Press: {key_press}", (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.imshow("Image", img)

            key = cv2.waitKey(10)
            # Press different key to collect different infomation
            for gs in gesture_string:
                if key == ord(gs):
                    key_press = gs
                    if not os.path.exists(f"{folder}/{gs}"):
                        print("Create corresponding data folder......:")
                        os.mkdir(f"{folder}/{gs}")
                    counter += 1
                    cv2.imwrite(
                        f'{folder}/{gs}/Image_{time.time()}.jpg', imgWhite)
                    cv2.waitKey(300)
            if key not in [ord(gs) for gs in gesture_string]:
                key_press = ""

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
