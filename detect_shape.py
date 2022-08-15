import cv2
img =cv2.imread("./img/geo.png")
img_contours = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 100, 150)
contours, hieranrchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        proxi_len = cv2.arcLength(cnt, True)
        verties = cv2.approxPolyDP(cnt, proxi_len * 0.01, True)  # 近似的頂點
        x, y, w, h = cv2.boundingRect(verties)
        cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)
        cv2.drawContours(img_contours, cnt, -1, (255, 0, 0), thickness=3)
        print(f"Detect: area={area}, # of vert={len(verties)}")
        if len(verties) == 3:
            cv2.putText(img_contours, "Triange", 
                org=(x, y-5), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.3, 
                color=(0, 255, 0)
            )
        elif len(verties) == 4:
            cv2.putText(img_contours, "Rectangle", 
                org=(x, y-5), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.3,
                color=(0, 255, 0)
            )
        else:            
            cv2.putText(img_contours, "Circle", 
                org=(x, y-5), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.3,
                color=(0, 255, 0)
            )


cv2.imshow("img", img)
cv2.imshow("canny", canny)
cv2.imshow("contours", img_contours)
cv2.waitKey(0)