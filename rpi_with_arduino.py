import time
import serial  # 導入serial庫
from detect_hands_cvzone import CvzoneHandDetector
import cvzone
import cv2
# ls /dev/tty*
port_name = '/dev/tty.usbserial-D307TEN6'  # Mac
# port_name = '/dev/ttyUSB0'  # Pi 4B


def detect_and_draw(frame, detector, is_detection_draw=True):

    drawed_img, hands = hand_detector.detect_object(
        img, is_detection_draw=True)

    fh, fw = img.shape[:2]
    img_center = (fw//2, fh//2)
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
        cv2.circle(
            img, hand['center'],
            radius=3, color=(255, 0, 255))
        cv2.arrowedLine(
            img, img_center, hand['center'],
            color=[0, 0, 255], thickness=2)
    return drawed_img, hands


def draw_bias(frame, hand_data):
    fh, fw = frame.shape[:2]
    img_center = (fw//2, fh//2)
    depth = int(hand_data["depth_in_cm"])
    x, y, w, h = hand_data['bbox']
    cvzone.putTextRect(
        drawed_img, f"x:{hand_data['bias_angle_x']}, y:{hand_data['bias_angle_y']}",
        (x+w-10, y)
    )
    cvzone.putTextRect(
        drawed_img, f'{depth} cm', (x, y+h+10)
    )
    cv2.circle(
        img, hand_data['center'],
        radius=3, color=(255, 0, 255))
    cv2.arrowedLine(
        img, img_center, hand_data['center'],
        color=[0, 0, 255], thickness=2)


def init_usb_serial_port(port_name, bord_rate=9600, timeout=0.3):
    # 打開端口，每一秒返回一個消息
    return serial.Serial(port_name, bord_rate, timeout=timeout)


if __name__ == "__main__":
    hand_detector = CvzoneHandDetector()
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Arduino Board
    is_response_got = True
    serial_port = init_usb_serial_port(port_name)
    # Loop
    while True:
        success, img = cap.read()
        # Send to Arduino Board
        if is_response_got:
            drawed_img, hands = detect_and_draw(img, hand_detector)
            if hands:
                dx = hands[0]['bias_angle_x']
                dy = hands[0]['bias_angle_y']
                data_in_string = f"{dx},{dy},"
                print(f"Send Data...., {data_in_string}")
                data_to_be_send = str.encode(data_in_string)
                serial_port.write(data_to_be_send)  # 寫s字符
                is_response_got = False
        else:
            response = serial_port.readall()  # 用response讀取端口的返回值
            if response:
                is_response_got = True
                print("Response:\n", response.decode('UTF8'))  # 進行打印

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            serial_port.close()
            break
