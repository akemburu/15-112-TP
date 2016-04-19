import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")

img = cv2.imread('thumb_DPP_0570_1024.jpg')
files = []

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces: 
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_eyes_gray = gray[y:y+(4.9/10.0)*h, x:x+w]
    roi_eyes_color = img[y:y+(4.9/10.0)*h, x:x+w]
    roi_mouth_gray = gray[y+(6.5/10.0)*h:y+h, x:x+w]
    roi_mouth_color = img[y+(6.5/10.0)*h:y+h, x:x+w]
    #different regions of interest for the eyes and the mouth
    #because they are located on different place of the face
    eyes = eye_cascade.detectMultiScale(roi_eyes_gray)
    mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_eyes_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        cropEyes = roi_eyes_color[ey:ey+eh, ex:ex+ew]
        if "eye_1.jpg" in files: 
            cv2.imwrite("eye_2.jpg",cropEyes)
        else: 
            cv2.imwrite("eye_1.jpg", cropEyes)
            files.append("eye_1.jpg")
    for (mx, my, mw, mh) in mouth: 
        cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255), 2)
        cropMouth = roi_mouth_color[my:my+mh, mx:mx+mw]
        cv2.imwrite("mouth.jpg",cropMouth)


cv2.imshow('img', img)
newImage = cv2.imread("eye_1.jpg")
holla = cv2.imread("eye_2.jpg")
cv2.imshow("newImage",newImage)
otherImage = cv2.imread("mouth.jpg")
cv2.imshow("otherImage", otherImage)
cv2.imshow("newImage", newImage)
cv2.imshow("holla", holla)

k = cv2.waitKey(0) & 0xff
if k == 27:  
    cap.release()
    cv2.destroyAllWindows()  