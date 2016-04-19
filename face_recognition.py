import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#sources: http://alereimondo.no-ip.org/OpenCV/34

cap = cv2.VideoCapture(0)
#finds the webcam that this program will use 

while True: 
    ret, img = cap.read()
    #reads the image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #converts it to a grayscale image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #uses the grayscale and the cascade to detect a face 
    for (x,y,w,h) in faces: 
        #using the region where a face was defined 
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #region of interests, based on approximate location of eyes and nose 
        roi_eyes_gray = gray[y:y+(5/10.0)*h, x:x+w]
        roi_eyes_color = img[y:y+(5/10.0)*h, x:x+w]
        roi_mouth_gray = gray[y+(6.5/10.0)*h:y+h, x:x+w]
        roi_mouth_color = img[y+(6.5/10.0)*h:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_eyes_gray)
        mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_eyes_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        for (mx, my, mw, mh) in mouth: 
            cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27: 
        #the "ESC" key will break the while loop 
        break 

cap.release()
cv2.destroyAllWindows()  

