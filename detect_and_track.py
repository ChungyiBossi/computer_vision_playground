from detect_object_dnn import (
    load_coco_name,
    load_object_detection_model,
    detect_object_NMS
)
from track_single_object import (
    create_single_object_tracker_model
)
import cv2


def detect_and_track_object(
    detector, class_names, maximum_of_trackers=3,
    video_capture=0, threshold=0.45, nms_threshold=0.2
):
    tracking = False                    # 設定 False 表示尚未開始追蹤
    cap = cv2.VideoCapture(video_capture)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    trackers = list()

    while True:
        # capture
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame = cv2.resize(frame, (540, 300))  # 縮小尺寸，加快速度
        keyName = cv2.waitKey(1)

        if keyName == ord('q'):
            print("Stop tracking")
            break

        if tracking:
            for k, tracker in enumerate(trackers):
                success, point = tracker.update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
                # 需處理全部失敗的狀況
                if success:
                    p1 = [int(point[0]), int(point[1])]
                    p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
                    cv2.rectangle(
                        img=frame,
                        pt1=p1,
                        pt2=p2,
                        color=(0, 0, 255),
                        thickness=3
                    )
                    # 根據座標，繪製四邊形，框住要追蹤的物件
        else:
            frame, classIds, bbox, confs, nms_indices = detect_object_NMS(
                object_detection_model=detector,
                frame=frame,
                class_names=class_names,
                detect_threshold=threshold,
                nms_threshold=nms_threshold
            )

            if len(nms_indices):
                top_n = sorted([
                    (class_names[classIds[i]-1], bbox[i], confs[i])
                    for i in nms_indices
                ], key=lambda x: x[2])

                print(top_n)

                for idx, target in enumerate(top_n[:maximum_of_trackers]):
                    trackers.append(create_single_object_tracker_model())
                    trackers[idx].init(frame, tuple(target[1]))

                tracking = True

        cv2.imshow('Tracker', frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = load_object_detection_model()
    class_names = load_coco_name()
    detect_and_track_object(
        detector, class_names,
        video_capture=0,
        threshold=0.45,
        nms_threshold=0.2
    )
