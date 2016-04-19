import numpy as np 
import cv2 
import math 
import csv

#each eye and mouth will be stores as objects from eye class 
#this way they can be accessed and used from inner methods  
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#from: http://alereimondo.no-ip.org/OpenCV/34

class facialFeatures(object): 
    def calculateRoiPixels(self): 
        roi = cv2.imread(self.croppedImage)
        allPixelMinimum = np.array([0,0,0], np.uint8)
        allPixelMaximum = np.array([255,255,255], np.uint8)
        dstAll = cv2.inRange(roi, allPixelMinimum, allPixelMaximum)
        pixels = cv2.countNonZero(dstAll)
        return pixels 

class eyes(facialFeatures): 
    files = [] 
    def __init__(self, x,y,width, height,roi): 
        self.x = x
        self.y = y
        self.width = width 
        self.height = height
        self.image = roi 
        self.croppedImage = cropImage(self)
        super().__init__()

    def cropImage(self): 
        cropEyes = self.image[self.x:self.y, self.width:self.height]
        if "eye_1" in eyes.files: 
            cv2.imwrite("eye_2.jpg",cropEyes)
            return "eye_2.jpg"
        else: 
            cv2.imwrite("eye_1.jpg", cropEyes)
            eyes.files.append("eye_1.jpg")
            return "eye_1.jpg"

        def calculateOpenEyePixels(self): 
        eyeImg = cv2.imread(self.croppedImage)
        whitePixel = np.array([255,255,255],np.uint8)
        dstWhite = cv2.inRange(eyeImg, whitePixel, whitePixel)
        noWhite = cv2.countNonZero(dstWhite)
        return noWhite

    def calculatePupilSize(self): 
        eyeImg = cv2.imread(self.croppedImage)
        gray = cv2.cvtColor(eyeImg, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT, 1.2, 40)
        if circles != None: 
            circles = np.round(circles[0, :]).astype("int")
            x,y,r = circles 
            return (r ** 2) * math.pi
        else: 
            return 0 


class mouth(facialFeatures): 
    def __init__(self, x,y,width,height,roi):
         self.x = x 
         self.y = y 
         self.width = width 
         self.height = height
         self.image = roi 
         self.croppedMouth = cropImage(self)

    def cropImage(self):
        mouthImg = self.image[self.x:self.y, self.width:self.height]
        cv2.imwrite("mouth.jpg", mouthImg)
        return "mouth.jpg" 

    def calculateOpenMouthPixels(self): 
        #use a grayscale image 
        mouthImg = cv2.imread(self.croppedMouth)
        whitePixel = np.array([255,255,255],np.uint8)
        dstWhite = cv2.inRange(eyeImg, whitePixel, whitePixel)
        noWhite = cv2.countNonZero(dstWhite)
        return noWhite

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
            eyes(ex,ey,ew,eh,roi_eyes_color)

        for (mx, my, mw, mh) in mouth: 
            cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255),2)
            mouth(mx,my,mw,mh,roi_mouth_color)

#here starts the KNN Algorithm 

def calculateAreas(): 
    eyeArea = 0 
    totalEyeArea = 0 
    for eye in eyes: 
        eyeArea += eye.calculateOpenEyePixels() + eye.calculatePupilSize()
        totalEyeArea += eye.calculateRoiPixels()
    for mouth in mouth: 
        mouthArea = mouth.calculateOpenMouthPixels()
        totalMouthArea = mouth.calculateRoiPixels()
    return (eyeArea/totalEyeArea, mouthArea/totalMouthArea)


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

#def createCSVFile





