import numpy as np 
import cv2 
import math 
from Tkinter import *
import csv

#each eye and mouth will be stores as objects from eye class 
#this way they can be accessed and used from inner methods  
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#from: http://alereimondo.no-ip.org/OpenCV/34

def captureFrame():
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
    findROIs() 

def findROIs(): 
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
        eyes = eye_cascade.detectMultiScale(roi_eyes_gray)
        mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)
        if len(eyes) == 2: 
            ex, ey, ew, eh = eyes[0]
            eye1 = eyesClass(ex,ey,ew,eh)
            ex2, ey2, ew2, eh2 = eyes[1]
            eye2 = eyesClass(ex2, ey2, ew2,eh2)
            cv2.rectangle(roi_eyes_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
            cv2.rectangle(roi_eyes_color,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
        else: 
            print("eye error")
            print(eyes)
            #tk.messageBox.showwarning("Error", "Please retake the image, following the instructions")
            #captureFrame()
        if len(mouth) == 1:
            mx,my,mw,mh = mouth[0]
            cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255),2)
            mouth1 = mouthClass(mx,my,mw,mh)
        else: 
            print("mouth error")
            print(mouth)
            #tk.messageBox.showwarning("Error", "Please retake the image, following the instructions")
            #captureFrame()
    if len(eyes) == 2 and len(mouth) == 1: 
        calculateAreas(eye1,eye2,mouth1)


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
        #totalSpace = eyeImg.calculateRoiPixels()
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
        #totalSpace = mouthImg.calculateRoiPixels()
        return totalSpace - white 

def calculateAreas(eye1,eye2,mouth1): 
    eyeArea = 0 
    eyeArea += eye1.calculateOpenEyePixels() + eye1.calculatePupilSize()
    eyeArea += eye2.calculateOpenEyePixels() + eye2.calculatePupilSize()
    #print(eye1.calculateOpenEyePixels(),eye1.calculatePupilSize(),eye2.calculateOpenEyePixels(),eye2.calculatePupilSize())
    totalEyeArea = eye1.calculateRoiPixels() + eye2.calculateRoiPixels()
    #print(eye1.calculateRoiPixels(), eye2.calculateRoiPixels())
    mouthArea = mouth1.calculateOpenMouthPixels()
    totalMouthArea = mouth1.calculateRoiPixels()
    #print(mouthArea, totalMouthArea)
    #print(eyeArea/(totalEyeArea*1.0), mouthArea/(1.0*totalMouthArea))
    return (eyeArea/totalEyeArea, mouthArea/totalMouthArea)

###############################################################################
#here starts the KNN Algorithm 

def makeDataSet(filename): 
    #open the file using filename in this line 
    file = open(filename)
    officialFile = csv.reader(file)
    dataSet = [] 
    for row in officialFile:
        x = float(officialFile[row][0])
        y = float(officialFile[row][1])
        state = officialFile[row][2]
        dataSet.append([x,y,state])
    #check to make sure that every number is a decimal 
    return dataSet
 
def calculateDistance(x,y,x1,y1):
    distance = (x-x1)**2 + (y-y1)**2
    return math.sqrt(distance)
    
def getKey(item):
    return item[1]

def neighbors(k,filename = "data.csv"):
    #where x defines size of eyes
    #where y defines size of mouth 
    x, y = calculateAreas() 
    dataSet = makeDataSet(filename)
    allDists = [] 
    for i in range(len(dataSet)):
        x1 = dataSet[i][0]
        y1 = dataSet[i][1]
        distance = calculateDistance(x,y,x1,y1)
        allDists.append((dataSet[i], distance))
    allDists.sort(key = getItem)
    neighbors = []
    for j in range(k):
        neighbors.append(allDists[j][0])
    return neighbors

def getMaxKey(counts): 
    values = list(counts.values())
    keys = list(counts.keys())
    maxValue = max(values)
    index = values.index(values)
    return k[index]

def classifyState(neighbors):
    grouping = dict()
    for row in range(len(neighbors)): 
        state = neighbors[row][-1]
        if state in grouping: 
            grouping[state] += 1 
        else: 
            grouping[state] = 1 
    classification = getMaxKey(grouping)
    return classification

captureFrame()




