import cv2
import numpy

img = cv2.imread("./data/img/4.d6206092.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 1080)
cv2.createTrackbar("HueMin", "TrackBar", 0, 179, lambda x: 0)
cv2.createTrackbar("HueMax", "TrackBar", 179, 179, lambda x: 0)
cv2.createTrackbar("SatMin", "TrackBar", 0, 179, lambda x: 0)
cv2.createTrackbar("SatMax", "TrackBar", 179, 179, lambda x: 0)
cv2.createTrackbar("ValMin", "TrackBar", 0, 179, lambda x: 0)
cv2.createTrackbar("ValMax", "TrackBar", 179, 179, lambda x: 0)

while cv2.waitKey(1) != ord("q"):
    h_min = cv2.getTrackbarPos("HueMin", "TrackBar")
    h_max = cv2.getTrackbarPos("HueMax", "TrackBar")
    s_min = cv2.getTrackbarPos("SatMin", "TrackBar")
    s_max = cv2.getTrackbarPos("SatMax", "TrackBar")
    v_min = cv2.getTrackbarPos("ValMin", "TrackBar")
    v_max = cv2.getTrackbarPos("ValMax", "TrackBar")
    lower = numpy.array([h_min, s_min, v_min])
    upper = numpy.array([h_max, s_max, v_max])

    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img", img)
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
