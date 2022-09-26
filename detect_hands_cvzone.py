from object_detector_interface import ObjectDetector
from cvzone.HandTrackingModule import HandDetector
import cvzone
import cv2
import math
import numpy as np


class CvzoneHandDetector(ObjectDetector):
    def __init__(self):
        self.model = self.load_model(detectionCon=0.8, maxHands=2)
        self.class_names = self.load_class_name()

    def load_model(self, detectionCon, maxHands):
        return HandDetector(detectionCon, maxHands)

    def load_class_name(self):
        # Not used
        return ['LeftHand', "RightHand"]

    def detect_object(self, frame, is_detection_draw=True, **kwargs):
        if is_detection_draw:
            hands_data, drawed_img = self.model.findHands(
                frame, draw=is_detection_draw)
        else:
            hands_data = self.model.findHands(
                frame, draw=is_detection_draw)
            drawed_img = frame

        fh, fw = frame.shape[:2]
        for hand in hands_data:
            lmList = hand['lmList']
            d = self.hand_depth(lmList)
            delta_pixel = \
                (hand["center"][0] - fw//2,  # x
                 hand["center"][1] - fh//2)  # y

            hand["depth_in_cm"] = d
            hand["bias_angle_x"] = \
                -1 * self.caculate_delta_angle(delta_pixel[0], d)  # 成像左右相反
            hand["bias_angle_y"] = self.caculate_delta_angle(delta_pixel[1], d)

        return drawed_img, hands_data

    def hand_depth(self, handlandmark):
        # focus:w = depth:W , depth = focus*W/w = F(w)
        x1, y1, z1 = handlandmark[5]
        x2, y2, z2 = handlandmark[17]
        w = int(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2))

        # TODO: How to implement distance calculation in 3d space?
        # Z-axis is not precision enough to use (in same distance, rotate hand, variable 3d-distance in different rotate angle)
        # w_3d = int(((y2 - y1) ** 2 + (x2 - x1) ** 2 + (z2 - z1) ** 2) ** (1/2))
        # print("\tDist 2D in pixel: ", w)
        # print("\tDist 3D in pixel: ", w_3d)

        A, B, C = self.fit_depth_poly()
        depth = A * (w**2) + B * w + C
        return depth

    def caculate_delta_angle(self, delta_pixel, object_depth, round_deg=2):
        delta_cm = self.pixel_to_cm(delta_pixel)
        return round(math.degrees(math.atan2(delta_cm, object_depth)), round_deg)

    def pixel_to_cm(self, cm):
        return cm * 0.02645833

    def fit_depth_poly(self):
        # Manually meansure
        # Find Function
        image_distance_in_pixel = [
            300, 245, 200, 170, 145, 130,
            112, 103, 93, 87, 80, 75,
            70, 67, 62, 59, 57
        ]
        real_distance_in_CM = [
            20, 25, 30, 35, 40, 45,
            50, 55, 60, 65, 70, 75,
            80, 85, 90, 95, 100
        ]
        return np.polyfit(
            image_distance_in_pixel,
            real_distance_in_CM, 2)  # y = Ax^2 + Bx + C


if __name__ == "__main__":
    hand_detector = CvzoneHandDetector()

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Loop
    while True:
        success, img = cap.read()
        drawed_img, hands = hand_detector.detect_object(
            img, is_detection_draw=False)
        for hand in hands:
            depth = int(hand["depth_in_cm"])
            x, y, w, h = hand['bbox']
            cvzone.putTextRect(
                drawed_img, f"x:{hand['bias_angle_x']}, y:{hand['bias_angle_y']}",
                (x+w-10, y)
            )
            cvzone.putTextRect(
                drawed_img, f'{depth} cm', (x, y+h+10)
            )
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
