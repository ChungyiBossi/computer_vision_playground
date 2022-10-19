import time
from urllib import response
import serial  # 導入serial庫
from detect_hands_cvzone import CvzoneHandDetector
import cv2
# ls /dev/tty*
port_name = '/dev/tty.usbserial-D307TEN6'  # Mac
# port_name = '/dev/ttyUSB0'  # Pi 4B


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
            drawed_img, hands = hand_detector.detect_object(img)
            if hands:
                dx = hands[0]['bias_angle_x']
                dy = hands[0]['bias_angle_y']
                data_in_string = f"{dx},{dy},"
                print(f"Send Data...., {data_in_string}")
                data_to_be_send = str.encode(data_in_string)
                serial_port.write(data_to_be_send)  # 寫s字符
                is_response_got = False
        else:
            # s_time = time.time()
            # response = serial_port.readall()  # 會有0.3sec的delay
            resp = serial_port.read(
                serial_port.in_waiting)  # 限定讀取的byte
            # print(f"Read time: {time.time() - s_time}")
            if resp:
                is_response_got = True
                print("Response:\n", resp.decode('UTF8'))  # 進行打印

        for hand in hands:
            img = hand_detector.draw_bias(img, hand, fixed_width//480)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            serial_port.close()
            break
