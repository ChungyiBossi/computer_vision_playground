import cv2
import glob

root_dir = "./people/test/"
result_dir = "./people/test_preprocess/"


def read_images(folder_path):
    file_list = []
    for filename in glob.iglob(folder_path + '**/*.jpg', recursive=True):
        #  print(filename)
        file_list.append(filename)
    return file_list


for file_path in read_images(root_dir):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faceRect = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=18,
        minSize=(32, 32)
    )

    for idx, (x, y, w, h) in enumerate(faceRect):
        # cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 2)
        new_file_name = file_path.replace(
            root_dir, result_dir).replace(".jpg", f"_{idx}.jpg")
        print("new_file_name", new_file_name)
        cv2.imwrite(new_file_name, img[y:y+h, x:x+w])
        cv2.imshow("facial recognition", img[y:y+h, x:x+w])
        cv2.waitKey(500)
    print(file_path)
    cv2.imshow("facial recognition", img)
    cv2.waitKey(500)
