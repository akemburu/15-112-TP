import numpy as np 
import cv2 
import sys

cap = cv2.VideoCapture(0)

while True: 
    ret, img = cap.read() 
    k = cv2.waitKey(1) & 0xff
    cv2.imshow('img', img)
    if k == 27: 
    	cap.release()
    	cv2.destroyAllWindows()  
        break 
    elif k == ord('p'): 
        cv2.imwrite("test.jpg", img) 
        cap.release()
        cv2.destroyAllWindows() 
        break
         



