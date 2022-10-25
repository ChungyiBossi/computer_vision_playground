import tensorflow as tf
import cv2
import glob
import numpy as np
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model(
    './keras_models/keras_model.h5', compile=False)  # 載入模型
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)          # 設定資料陣列


def face_detection(img_data, detector=face_detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_result = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=18,
        minSize=(32, 32)
    )
    return detect_result


def show_text(img, text):      # 建立顯示文字的函式
    org = (0, 50)     # 文字位置
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 文字字型
    fontScale = 1                        # 文字尺寸
    color = (255, 255, 255)                # 顏色
    thickness = 2                        # 文字外框線條粗細
    lineType = cv2.LINE_AA               # 外框線條樣式
    cv2.putText(img, text, org, fontFace, fontScale,
                color, thickness, lineType)  # 放入文字


def read_images(folder_path):
    file_list = []
    for filename in glob.iglob(folder_path + '**/*.jpg', recursive=True):
        #  print(filename)
        file_list.append(filename)
    return file_list


# get img file
label_name = ["Song", "Chi", "Nu"]
preprocessed_dir = "./people/test_preprocess/"
for file_path in read_images(folder_path=preprocessed_dir):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (398, 224))[0:224, 80:304]
    img_array = np.asarray([img])
    normalize_img_array = (img_array.astype(np.float32) / 127.0) - 1
    prediction = model.predict(normalize_img_array)[0]

    print(prediction)
    show_text(img, label_name[np.argmax(prediction)])
    cv2.imshow("Prediction", img)
    cv2.waitKey(2000)
