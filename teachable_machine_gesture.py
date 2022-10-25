from operator import ge
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collect_gesture_data import capture_hand_part
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("./gesture_model/keras_model.h5",
                        "./gesture_model/labels.txt")

classifier.model.summary()

# Save YAML
yaml = classifier.model.to_yaml()
with open("./gesture_model/model_config.yaml", "w") as cfg:
    cfg.write(yaml)
offset = 20
imgSize = 300
gesture_string = classifier.list_labels

while True:
    success, img = cap.read()
    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = capture_hand_part(img, hand)

            prediction, index = classifier.getPrediction(
                imgWhite, draw=False)

            prob = round(prediction[index], 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{gesture_string[index]}-{prob:.2f}", (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)

            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("ImageOutput", imgOutput)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
