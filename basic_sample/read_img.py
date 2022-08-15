import cv2
import numpy
img = cv2.imread("SOMEIMAGE.jpg")

# cv2.resize(img, (480, 360))
# img = cv2.resize(img, (0,0), fx=5, fy=5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(img, (9, 9), 7)  # need background knowledge
canny = cv2.Canny(img, 150, 200)  # need background knowledge

dilate_kernel = numpy.ones((5, 5), numpy.uint8)
# need background knowledge # 變粗線條
dilate = cv2.dilate(canny, dilate_kernel, iterations=1)

erode_kernel = numpy.ones((3, 3), numpy.uint8)
erode = cv2.erode(dilate, erode_kernel, iterations=1)

cv2.imshow('gray', gray)
cv2.imshow('gaussian', gaussian)
cv2.imshow('canny', canny)
cv2.imshow('dilate', dilate)
cv2.imshow('erode', erode)
cv2.imshow('color_img', img)
cv2.waitKey(0)
