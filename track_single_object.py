import cv2
import time

# def create_multi_object_tracker_model():
#     return cv2.legacy.MultiTracker_create()


class ObjectTracker():
    def __init__(self, max_untracked_frame_endurence=30, is_arrow_from_center_drawed=False):
        self.tracker = self.create_single_object_tracker_model()
        self.is_arrow_from_center_drawed = is_arrow_from_center_drawed
        self.is_tracking = False
        self.num_of_untracked_frame = 0
        self.object_dimension = -1
        self.max_untracked_frame_endurence = max_untracked_frame_endurence
        self.tracked_object_name = None
        self.tracked_object_point = (-1, -1)

    @staticmethod
    def create_single_object_tracker_model():
        # tracker = cv2.TrackerCSRT_create()  # 創建追蹤器
        tracker = cv2.TrackerKCF_create()
        return tracker

    def init(self, frame, area, object_class):
        self.tracker.init(frame, area)
        self.num_of_untracked_frame = 0
        self.is_tracking = True
        self.tracked_object_name = object_class

    def update(self, frame):
        # 追蹤成功後，不斷回傳左上座標和高寬
        success, point = self.tracker.update(frame)
        if success:
            self.num_of_untracked_frame = 0
            x, y, w, h = point
            self.tracked_object_point = (x + w//2, y + h//2)
            if self.is_arrow_from_center_drawed:
                self.draw_dimension_arrow(frame)
                self.object_dimension = self.check_target_diemsion(
                    self.get_frame_center(frame), self.tracked_object_point)
        else:
            self.num_of_untracked_frame += 1
            # TODO: refactor
            self.object_dimension = -1
            point = (0, 0, 0, 0)
            if self.num_of_untracked_frame > self.max_untracked_frame_endurence:
                self.is_tracking = False
        return point

    def get_frame_center(self, frame):
        fh, fw = frame.shape[:2]
        return (fw//2, fh//2)

    def check_target_diemsion(self, center_point, tracking_point):
        cx, cy = center_point
        tx, ty = tracking_point
        vector = (tx-cx, ty-cy)
        dimension = -1
        if vector[0] > 0 and vector[1] > 0:
            dimension = 4
        elif vector[0] > 0 and vector[1] < 0:
            dimension = 1
        elif vector[0] < 0 and vector[1] > 0:
            dimension = 3
        else:  # vector[0] < 0 and vector[1] <0
            dimension = 2

        return dimension

    def draw_tracking_object(self, frame, point, color=(0, 0, 255)):
        # 根據座標，繪製四邊形，框住要追蹤的物件
        p1 = [int(point[0]), int(point[1])]
        p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
        cv2.rectangle(
            img=frame,
            pt1=p1,
            pt2=p2,
            color=color,
            thickness=3
        )
        cv2.putText(
            frame,
            f"{self.tracked_object_name}",
            (p1[0]+5, p1[1]+10),
            cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1
        )

        if self.is_arrow_from_center_drawed:
            if self.object_dimension != -1:
                cv2.putText(
                    frame,
                    f"{self.object_dimension}",
                    (p2[0]-15, p2[1]-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                )

    def draw_dimension_arrow(self, frame, color=(255, 0, 0)):
        point_to = self.tracked_object_point
        cv2.circle(frame, point_to, radius=3, color=(255, 0, 255))
        cv2.arrowedLine(frame, self.get_frame_center(
            frame), point_to, color, thickness=2)


def track_object(video_capture=0):
    tracking = False                    # 設定 False 表示尚未開始追蹤
    cap = cv2.VideoCapture(video_capture)
    time.sleep(1)
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
            tracker = ObjectTracker(is_arrow_from_center_drawed=True)
            tracker.init(frame, area, "Target")    # 初始化追蹤器
            tracking = True              # 設定可以開始追蹤

        if tracking:
            point = tracker.update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
            if tracker.is_tracking:
                tracker.draw_tracking_object(frame, point)
            else:
                tracker = None
                tracking = False
                print("Update fail...")
        else:
            print("Not Tracking")

        cv2.imshow('Tracker', frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_object()
