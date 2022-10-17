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
        drawed_img = draw_bias(drawed_img, hand)
    return drawed_img, hands


def draw_bias(frame, hand_data, scale=1):
    fh, fw = frame.shape[:2]
    img_center = (fw//2, fh//2)
    depth = int(hand_data["depth_in_cm"])
    x, y, w, h = hand_data['bbox']
    cvzone.putTextRect(
        frame, f"x:{hand_data['bias_angle_x']}, y:{hand_data['bias_angle_y']}",
        (x+w-10, y),
        scale=scale
    )
    cvzone.putTextRect(
        frame, f'{depth} cm', (x, y+h+10),
        scale=scale
    )
    cv2.circle(
        frame, hand_data['center'],
        radius=3, color=(255, 0, 255))
    cv2.arrowedLine(
        frame, img_center, hand_data['center'],
        color=[0, 0, 255], thickness=2)

    return frame


def init_usb_serial_port(port_name, bord_rate=9600, timeout=0.3):
    # 打開端口，每一秒返回一個消息
    return serial.Serial(port_name, bord_rate, timeout=timeout)


if __name__ == "__main__":
    hand_detector = CvzoneHandDetector()
    fixed_width = 480
    fixed_height = 360
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fixed_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fixed_height)
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
            else:
                # Draw previous result
                for hand in hands:
                    img = draw_bias(img, hand, fixed_width//480)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            serial_port.close()
            break
