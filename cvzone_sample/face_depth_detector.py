import cv2
import cvzone
# from cvzone.FaceDetectionModule import FaceDetector()
from cvzone.FaceMeshModule import FaceMeshDetector

detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # Drawing
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # # Finding the Focal Length
        # d = 50  # 你的距離
        # f = (w*d)/W
        # print("Focus: ", f)

        # Finding distance
        f = 1500
        d = (W * f) / w
        print("Distance: ", d)

        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),  # 10 是頭頂
                           scale=2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
