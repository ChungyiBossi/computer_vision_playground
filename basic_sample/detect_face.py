import cv2
img = cv2.imread("./img/4.d6206092.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faceRect = detector.detectMultiScale(
    gray,
    scaleFactor=1.08,
    minNeighbors=15,
    minSize=(32, 32)
)

for x, y, w, h in faceRect:
    cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.imshow("face", gray)
cv2.waitKey(0)