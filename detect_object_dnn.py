import numpy as np
import cv2
from object_detector_interface import ObjectDetector


class CocoDetector(ObjectDetector):
    def __init__(
        self,
        config_path='./tf_models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
        weight_path='./tf_models/frozen_inference_graph.pb',
        class_name_path='./tf_models/coco.names'
    ):

        self.model = self.load_model(config_path, weight_path)
        self.class_names = self.load_class_name(class_name_path)

    def load_model(self, config_path, weight_path):
        # Object detection model
        net = cv2.dnn_DetectionModel(weight_path, config_path)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        return net

    def load_class_name(self, class_name_path):
        with open(class_name_path, 'r') as f:
            classNames = f.read().split('\n')
        return classNames

    def detect_object(self, frame, detect_threshold, nms_threshold, is_detection_draw=True):
        # detecting.....
        classIds, confs, bbox = self.model.detect(
            frame, confThreshold=detect_threshold)

        class_names = [self.class_names[index-1].upper()
                       for index in classIds]
        # if object detected
        if len(classIds):
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            # 藉 NMS 取有效最大的bbox
            # https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-non-maximum-suppression-nms-aa70c45adffa
            indices = cv2.dnn.NMSBoxes(
                bbox, confs, detect_threshold, nms_threshold)

            if is_detection_draw:
                print("Draw detection bounding box.")
                for i in indices:
                    box = bbox[i]
                    class_name = class_names[i]
                    conf = round(confs[i], 2)

                    print(
                        f"Detect # {i}: class_name:{class_name}, conf:{conf}")
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(frame, (x, y), (x+w, h+y),
                                  color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        frame,
                        f"{class_name} {conf}",
                        (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
                    )
            return frame, class_names, bbox, confs, indices
        else:
            return frame, [], [], [], []


def detect_object_bounding_boxes(threshold=0.45, nms_threshold=0.2, video_capture=0):
    cap = cv2.VideoCapture(video_capture)
    detector = CocoDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Cannot receive frame")
            break
        img, classIds, bbox, confs, indices = \
            detector.detect_object(img, threshold, nms_threshold)
        cv2.imshow("Output", img)
        keyName = cv2.waitKey(30)
        if keyName == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_object_bounding_boxes()
