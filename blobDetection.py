import numpy as np 
import cv2 

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")

img = cv2.imread("thumb_DPP_0570_1024.jpg")
#this line of code stores the image read by openCV in the img variable
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#stores the grayscale of img by converting the color to gray 

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#detects faces using the face_cascade and the grayscale of the img 
#the second parameter represents scaleFactor
#the third parameter represents minimum neighbors 

for (x,y,w,h) in faces: 
    #extracts x,y, width, height of faces 
    #roi stands for the region of interest 
    roi_eyes_gray = gray[y:y+(4.9/10.0)*h, x:x+w]
    roi_eyes_color = img[y:y+(4.9/10.0)*h, x:x+w]
    roi_mouth_gray = gray[y+(6.5/10.0)*h:y+h, x:x+w]
    roi_mouth_color = img[y+(6.5/10.0)*h:y+h, x:x+w]
    #different regions of interest for the eyes and the mouth
    #because they are located on different place of the face
    eyes = eye_cascade.detectMultiScale(roi_eyes_gray)
    mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)

cv2.imshow("gray",gray)

k = cv2.waitKey(0) & 0xff
#this waitkey will wait until a key is pressed, 
if k == 27: 
    #escape key will destroy the window if the "ESC" is pressed
    cv2.destroyAllWindows()  

