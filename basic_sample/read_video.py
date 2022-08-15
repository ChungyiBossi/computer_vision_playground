import cv2

# cap = cv2.VideoCapture("./video/bmgqwviofdg95.mp4")
cap = cv2.VideoCapture(0) # capture camera

isGet = True
while isGet:
    isGet, frame = cap.read()
    if isGet:
        # frame = cv2.resize(frame, (0,0), fx=4, fy=4)
        cv2.imshow('frame', frame)

    if cv2.waitKey(15) == ord('q'):
        break