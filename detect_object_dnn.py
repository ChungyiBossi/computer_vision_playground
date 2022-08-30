import numpy as np
import cv2


def load_coco_name(class_file='./tf_models/coco.names'):
    with open(class_file, 'r') as f:
        classNames = f.read().split('\n')
    return classNames


def load_object_detection_model(
    config_path='./tf_models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
    weight_path='./tf_models/frozen_inference_graph.pb'
):
    # Object detection model
    net = cv2.dnn_DetectionModel(weight_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net


def detect_object_NMS(object_detection_model, frame, class_names, detect_threshold, nms_threshold, is_detection_draw=True):
    # detecting.....
    classIds, confs, bbox = object_detection_model.detect(
        frame, confThreshold=detect_threshold)
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
            for i in indices:
                box = bbox[i]
                conf = round(confs[i], 2)
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (x, y), (x+w, h+y),
                              color=(0, 255, 0), thickness=2)
                cv2.putText(
                    frame,
                    f"{class_names[classIds[i]-1].upper()} {conf}",
                    (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
                )
        return frame, classIds, bbox, confs, indices
    else:
        return frame, [], [], [], []


def detect_object_bounding_boxes(net, class_names=[], threshold=0.45, nms_threshold=0.2, video_capture=0):
    if class_names == []:
        class_names = load_coco_name()
    cap = cv2.VideoCapture(video_capture)
    while True:
        success, img = cap.read()
        if not success:
            print("Cannot receive frame")
            break
        img, classIds, bbox, confs, indices = \
            detect_object_NMS(net, img, class_names, threshold, nms_threshold)
        cv2.imshow("Output", img)
        keyName = cv2.waitKey(30)
        if keyName == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    net = load_object_detection_model()
    detect_object_bounding_boxes(net, load_coco_name())
