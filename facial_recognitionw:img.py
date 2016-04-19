import numpy as np 
import cv2 

#each eye and mouth will be stores as objects from eye class 
#this way they can be accessed and used from inner methods 
class eye(object): 
    def __init__(self, location): 
        print(location)
        self.eyeOnex = location[0]
        print(self.eyeOnex)
        self.eyeOney = location[1]
        print(self.eyeOney)
        self.eyeOnewidth = location[2]
        print self.eyeOnewidth
        self.eyeOneheight =  location[3]
        print self.eyeOneheight


class mouth(object): 
    def __init__(self,location):
        print location 
        self.mouthX = location[0]
        print self.mouthX
        self.mouthY = location[1]
        print self.mouthY 
        self.mouthWidth = location[2]
        print self.mouthWidth
        self.mouthHeight = location[3]
        print self.mouthHeight

    #include methods that determine amount of dark space of each object 
 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#from: http://alereimondo.no-ip.org/OpenCV/34

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

img = cv2.imread("test.jpg")
#this line of code stores the image read by openCV in the img variable
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#stores the grayscale of img by converting the color to gray 

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#detects faces using the face_cascade and the grayscale of the img 
#the second parameter represents scaleFactor
#the third parameter represents minimum neighbors 

for (x,y,w,h) in faces: 
    #extracts x,y, width, height of faces 
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #roi stands for the region of interest 
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
        #2 represents the thickness of the line 
    for (mx, my, mw, mh) in mouth: 
        cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255), 2)
        #2 represents the thickness of the line 

cv2.imshow('img', img)

print "faces detected: %d" % len(faces)
k = cv2.waitKey(0) & 0xff
#this waitkey will wait until a key is pressed, 
if k == 27: 
    #escape key will destroy the window if the "ESC" is pressed
	cv2.destroyAllWindows()  

print type(eyes)
print type(mouth)

def sendToClass(eyes, mouth):
    eyes = list(eyes)
    mouth = list(mouth)
    for eyeball in eyes: 
        if len(eyes) == 2:
            eye(eyeball)
    mouth(mouth)
    print "location of mouth: " + str(mouth)
    print "eyes detected: %d" % len(eyes)
    print "location of eyes: " + str(eyes)

sendToClass(eyes,mouth)






#def applyColorFilter() 

#def calculateDist(eyeSize, mouthSize, dataSet): 
    #updatedDict = {}
    #for key in dataSet:
        #baselineEyeSize, baselineMouthSize = dataSet[key]
        #dist = ((baselineEyeSize - eyeSize) ** 2) + ((baselineMouthSize - mouthSize)**2)
        #updatedDict[key] = dist
    #return updatedDict

#def classifyMood(eyeSize, mouthSize, dataSet, k): 
    #distDataSet = calcuateDist(eyeSize, mouthSize, dataSet) 
    #distData = [] 
    #for distance in distDataSet: 
        #distData += distance 
    #distData = distData[:k]
    #get the top k distData
    #topMoods = [] 
    #repeat =  


    #if user clicks next highest mood: 
        #dataSet - that mood
        #classifyMood(eyeSize, mouthSize, newDataSet, k)
        #stop drawing that button if mood == 1 
    #return the majority data set 





