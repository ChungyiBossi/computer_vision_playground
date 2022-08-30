import cv2


# def create_multi_object_tracker_model():
#     return cv2.legacy.MultiTracker_create()


def create_single_object_tracker_model():
    # tracker = cv2.TrackerCSRT_create()  # 創建追蹤器
    tracker = cv2.TrackerKCF_create()
    return tracker


def track_object(tracker, video_capture=0):
    tracking = False                    # 設定 False 表示尚未開始追蹤
    cap = cv2.VideoCapture(video_capture)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

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

        if keyName == ord('a'):
            # (min_x, min_y, w, h)
            area = cv2.selectROI('Tracker', frame,
                                 showCrosshair=False, fromCenter=False)
            tracker.init(frame, area)    # 初始化追蹤器
            tracking = True              # 設定可以開始追蹤

        if tracking:
            success, point = tracker.update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
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
                print("Update fail...")
        else:
            print("Not Tracking")

        cv2.imshow('Tracker', frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = create_single_object_tracker_model()
    track_object(tracker)
