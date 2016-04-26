import numpy as np
import cv2
import csv
import math

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")

def findROIs(filename): 
    img = cv2.imread(filename)
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
        global roi_eyes_gray
        global roi_eyes_color 
        global roi_mouth_gray 
        global roi_mouth_color 
        roi_eyes_gray = gray[y:y+(4.9/10.0)*h, x:x+w]
        roi_eyes_color = img[y:y+(4.9/10.0)*h, x:x+w]
        roi_mouth_gray = gray[y+(6.5/10.0)*h:y+h, x:x+w]
        roi_mouth_color = img[y+(6.5/10.0)*h:y+h, x:x+w]
        #different regions of interest for the eyes and the mouth
        #because they are located on different place of the face
        eyes = eye_cascade.detectMultiScale(roi_eyes_gray,1.1,6)
        mouth = mouth_cascade.detectMultiScale(roi_mouth_gray,1.1,50)
        if len(eyes) == 2: 
            ex, ey, ew, eh = eyes[0]
            eye1 = eyesClass(ex,ey,ew,eh)
            ex2, ey2, ew2, eh2 = eyes[1]
            eye2 = eyesClass(ex2, ey2, ew2,eh2)
        else: 
            print("eye error")
            #tk.messageBox.showwarning("Error", "Please retake the image, following the instructions")
            captureFrame()
        if len(mouth) == 1:
            mx,my,mw,mh = mouth[0]
            mouth1 = mouthClass(mx,my,mw,mh)
        else: 
            print("mouth error")
            #messageBox.showwarning("Error", "Please retake the image, following the instructions")
            captureFrame()
    if len(eyes) == 2 and len(mouth) == 1: 
        finalAnswer = neighbors(eye1,eye2,mouth1)
        return finalAnswer


class facialFeatures(object): 
    def calculateRoiPixels(self): 
        roi = cv2.imread(self.croppedImage)
        allPixelMinimum = np.array([0,0,0], np.uint8)
        allPixelMaximum = np.array([255,255,255], np.uint8)
        dstAll = cv2.inRange(roi, allPixelMinimum, allPixelMaximum)
        pixels = cv2.countNonZero(dstAll)
        return pixels 

class eyesClass(facialFeatures): 
    files = [] 
    def __init__(self, x,y,width, height): 
        self.x = x
        self.y = y
        self.width = width 
        self.height = height
        self.croppedImage = self.cropImage()
        print("Yay!!")

    def cropImage(self): 
        cropEyes = roi_eyes_color[self.y:self.y+self.height,self.x:self.x+self.width]
        if "eye_1.jpg" in eyesClass.files: 
            cv2.imwrite("eye_2.jpg",cropEyes)
            return "eye_2.jpg"
        else: 
            cv2.imwrite("eye_1.jpg", cropEyes)
            eyesClass.files.append("eye_1.jpg")
            return "eye_1.jpg"

    def calculateOpenEyePixels(self): 
        eyeImg = cv2.imread(self.croppedImage)
        hsv = cv2.cvtColor(eyeImg, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,82])
        upper_white = np.array([255,125,255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        equalizeWhiteSpace = cv2.equalizeHist(mask)
        return cv2.countNonZero(equalizeWhiteSpace)


    def calculatePupilSize(self): 
        eyeImg = cv2.imread(self.croppedImage)
        gray = cv2.cvtColor(eyeImg, cv2.COLOR_BGR2GRAY)
        row,col,_ = eyeImg.shape
        totalSpace = row * col
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        equalizePupil = cv2.equalizeHist(thresholded)
        white = cv2.countNonZero(equalizePupil)
        return totalSpace - white 

class mouthClass(facialFeatures): 
    def __init__(self, x,y,width,height):
        self.x = x 
        self.y = y 
        self.width = width 
        self.height = height
        self.croppedImage = self.cropImage()
        print("Yay!!")

    def cropImage(self):
        mouthImg = roi_mouth_color[self.y:self.y+self.height,self.x:self.x+self.width]
        cv2.imwrite("mouth.jpg", mouthImg)
        return "mouth.jpg" 

    def calculateOpenMouthPixels(self): 
        #use a grayscale image 
        mouthImg = cv2.imread(self.croppedImage)
        grayMouth = cv2.cvtColor(mouthImg, cv2.COLOR_BGR2GRAY)
        row,col,_ = mouthImg.shape
        totalSpace = row * col 
        _, thresholded = cv2.threshold(grayMouth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        equalizedMouth = cv2.equalizeHist(thresholded)
        white = cv2.countNonZero(equalizedMouth)
        return totalSpace - white 

def calculateAreas(eye1,eye2,mouth1): 
    eyeArea = 0 
    print(eye1.calculateOpenEyePixels())
    print(eye2.calculateOpenEyePixels())
    print(eye1.calculateRoiPixels(), eye2.calculateRoiPixels())
    print(mouth1.calculateOpenMouthPixels(), mouth1.calculateRoiPixels())
    eyeArea += eye1.calculateOpenEyePixels()
    eyeArea += eye2.calculateOpenEyePixels()
    totalEyeArea = eye1.calculateRoiPixels() + eye2.calculateRoiPixels()
    mouthArea = mouth1.calculateOpenMouthPixels()
    totalMouthArea = mouth1.calculateRoiPixels()
    return(eyeArea/(totalEyeArea*1.0), mouthArea/(1.0*totalMouthArea))


def makeDataSet(filename): 
    #open the file using filename in this line 
    dataSet = []
    with open(filename, 'rb') as csvfile:    
        data = list(csv.reader(csvfile)) 
        for row in range(len(data)):
            x = float(data[row][0])
            y = float(data[row][1])
            state = data[row][2]
            dataSet.append([x,y,state])
    return dataSet

def calculateDistance(x,y,x1,y1):
    distance = (x-x1)**2 + (y-y1)**2
    #reduce the weight of the mouth open variable
    return math.sqrt(distance)
    
def getKey(item):
    return item[1]

def neighbors(eye1,eye2,mouth1,k=5,filename = "data.csv"):
    #where x defines size of eyes
    #where y defines size of mouth 
    x,y = calculateAreas(eye1,eye2,mouth1)
    print(x,y)
    dataSet = makeDataSet(filename)
    allDists = [] 
    for i in range(len(dataSet)):
        distance = calculateDistance(x,y,dataSet[i][0],dataSet[i][1])
        allDists.append((dataSet[i], distance))
    newDst = sorted(allDists, key = getKey)
    print(newDst,"neighbors function")
    classification = []
    for j in range(k):
        group = newDst[j][0][2].strip()
        classification.append(group)
    #only has the 3-nearest nieghbors by classification 
    if classification.count("awake") > classification.count("sleepy"):
        print("AWAKE")
        return (x,y,"awake")
    elif classification.count("sleepy") > classification.count("awake"):
        print("SLEEPY")
        return (x,y,"sleepy")


findROIs("test.jpg")