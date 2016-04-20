import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")

img1 = cv2.imread("eye_1.jpg")
img2 = cv2.imread("eye_2.jpg")
img3 = cv2.imread("mouth.jpg")
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
#ray2 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imshow("thresholded",thresholded)

equ = cv2.equalizeHist(thresholded)
black = cv2.countNonZero(equ)
cv2.imshow("equ",equ)
cv2.imshow("img3", img3)


hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

lower_white = np.array([0,0,82])
upper_white = np.array([255,125,255])

mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(img1, img1, mask = mask)


equ2 = cv2.equalizeHist(mask)
white = cv2.countNonZero(equ2)

print(white)
#cv2.imshow("img1",img1)
#cv2.imshow('mask', mask)
#cv2.imshow("res", res)


k = cv2.waitKey(0) & 0xff
if k == 27: 
    cv2.destroyAllWindows() 