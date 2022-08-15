import cv2
# faces = face_cascade.detectMultiScale(img, scaleFactor, minNeighbors, flags, minSize, maxSize)
# 偵測並取出相關屬性
# img 來源影像，建議使用灰階影像
# scaleFactor 前後兩次掃瞄偵測畫面的比例係數，預設 1.1
# minNeighbors 構成檢測目標的相鄰矩形的最小個數，預設 3
# flags 通常不用設定，若設定 CV_HAAR_DO_CANNY_PRUNING 會使用 Canny 邊緣偵測，排除邊緣過多或過少的區域
# minSize, maxSize 限制目標區域的範圍，通常不用設定



def mosaic(img, x, y, cut_width, cut_height, level=15):
    area = img[y:y+cut_height, x:x+cut_width]
    w = int(cut_width/level)
    h = int(cut_height/level)
    mosaic = cv2.resize(
        area, (w,h), 
        interpolation=cv2.INTER_LINEAR
    )
    mosaic = cv2.resize(
        mosaic, (cut_width,cut_height), 
        interpolation=cv2.INTER_LINEAR
    )

    img[y:y+cut_height, x:x+cut_width] = mosaic
    return img


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame = cv2.resize(frame,(540,320))              # 縮小尺寸，避免尺寸過大導致效能不好
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # 將鏡頭影像轉換成灰階
        faces = face_cascade.detectMultiScale(gray)      # 偵測人臉
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   # 標記人臉
            frame = mosaic(frame, x, y, w, h)
        cv2.imshow('oxxostudio', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()






