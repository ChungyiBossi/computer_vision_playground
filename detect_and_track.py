from detect_object_dnn import (
    load_coco_name,
    load_object_detection_model,
    detect_object_NMS
)
from track_single_object import ObjectTracker
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
            for tracker in trackers:
                point = tracker.update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
                # 需處理全部失敗的狀況
                if tracker.is_tracking:
                    tracker.draw_tracking_object(frame, point)

            # 檢查是否全部都 untracked, 是的話則清空重新偵測
            check_trackers_tracking = False
            for tracker in trackers:
                if tracker.is_tracking:
                    check_trackers_tracking = True
                    break
            if not check_trackers_tracking:
                tracking = False
                trackers = list()

        else:
            frame, classIds, bbox, confs, nms_indices = detect_object_NMS(
                object_detection_model=detector,
                frame=frame,
                class_names=class_names,
                detect_threshold=threshold,
                nms_threshold=nms_threshold,
                is_detection_draw=False  # we drae when we tracking
            )
            if len(nms_indices):
                top_n = sorted([
                    (class_names[classIds[i]-1], bbox[i], confs[i])
                    for i in nms_indices
                ], key=lambda x: x[2])
                print(top_n)
                for target in top_n[:maximum_of_trackers]:
                    tracker = ObjectTracker()
                    tracker.init(frame, tuple(target[1]), target[0])
                    trackers.append(tracker)

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
