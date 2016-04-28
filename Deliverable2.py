import numpy as np 
import cv2 
import math 
from Tkinter import *
import tkMessageBox
import csv
from PIL import Image, ImageTk 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
import string 
import pickle

#each eye and mouth will be stores as objects from eye class 
#this way they can be accessed and used from inner methods  
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#from: http://alereimondo.no-ip.org/OpenCV/34

def init(data): 
    data.mode = "splashScreen"
    data.value = True 
    data.state = None 
    data.startButtonX = 900
    data.startButtonY = 500
    data.startButtonWidth = 200
    data.startButtonLength = 50
    data.continueButtonX = 500
    data.continueButtonY = 450
    data.continueButtonWidth = 200 
    data.continueButtonLength = 200
    data.Baymax = Image.open("Big_Hero_6_Baymax.gif")
    data.Baymax = data.Baymax.resize((1200,700), Image.ANTIALIAS)
    data.BaymaxImage = ImageTk.PhotoImage(data.Baymax)
    data.nextScreenX = 800
    data.nextScreenY = 200
    data.nextScreenWidth = 200
    data.nextScreenLength = 50 
    data.sleepyEmoji = Image.open("sleeping-face.gif")
    data.sleepyEmoji = data.sleepyEmoji.resize((300,300), Image.ANTIALIAS)
    data.sleepyEmojiImage = ImageTk.PhotoImage(data.sleepyEmoji)
    data.awakeEmoji = Image.open("relieved-face.gif")
    data.awakeEmoji = data.awakeEmoji.resize((300,300), Image.ANTIALIAS)
    data.awakeEmojiImage = ImageTk.PhotoImage(data.awakeEmoji)
    data.Camera = Image.open("Camera-icon.gif")
    data.Camera = data.Camera.resize((200,200), Image.ANTIALIAS)
    data.CameraImage = ImageTk.PhotoImage(data.Camera)    
    data.username = ""
    data.usernameBoxX = 950
    data.usernameBoxY = 300
    data.usernameBoxW = 200
    data.usernameBoxL = 50
    data.YesX = 400
    data.YesY = 600
    data.YesWidth = 150  
    data.YesLength = 50  
    data.NoX = 800 
    data.NoY = 600 
    data.NoWidth = 150
    data.NoLength = 50
    data.EnterX = 700
    data.EnterY = 500
    data.EnterWidth = 100
    data.EnterLength = 50
    data.sleepHours = "" 
    data.date = "" 
    data.results = None 
    data.eye = None
    data.mouth = None
    data.usernameBox = False
    data.sleepBoxX = 600
    data.sleepBoxY = 350
    data.sleepBoxWidth = 100
    data.sleepBoxLength = 50
    data.dateX = 600 
    data.dateY = 550
    data.dateWidth = 100
    data.dateLength = 50
    data.sleepBox = False 
    data.dateBox = False 
    data.accuracy = None
    data.baymax2 = Image.open("baymaxDoctor.gif")
    data.baymax2 = data.baymax2.resize((1200,700), Image.ANTIALIAS)
    data.baymax2Image = ImageTk.PhotoImage(data.baymax2) 
    data.graph = None
    data.logOutX = 1000
    data.logOutY = 100
    data.logOutWidth = 100
    data.logOutLength = 50
    data.error = False 

def mousePressed(event, data): 
    if data.mode == "splashScreen": 
        splashScreenMousedPressed(event,data)
    if data.mode == "instructionsScreen": 
        instructionsScreenMousedPressed(event, data) 
    if data.mode == "webcamScreen": 
        webcamScreenMousePressed(event,data)
    if data.mode == "resultScreen": 
        resultScreenMousePress(event,data) 
    if data.mode == "sleepScreen": 
        sleepScreenMousePressed(event,data) 
    if data.mode == "aboutScreen": 
        aboutScreenMousePressed(event,data)
    if data.mode == "graphScreen":
        graphScreenMousePressed(event,data)

def keyPressed(event, data): 
    if data.mode == "splashScreen": 
        splashScreenKeyPressed(event, data) 
    if data.mode == "instructionsScreen": pass 
    if data.mode == "webcamScreen": pass
    if data.mode == "resultScreen": pass
    if data.mode == "sleepScreen": 
        sleepScreenKeyPressed(event,data)  
    if data.mode == "aboutScreen": pass

def timerFired(data):
    pass

def redrawAll(canvas, data): 
    if data.mode == "splashScreen": 
        splashScreenReDrawAll(canvas, data)
    elif data.mode == "instructionsScreen": 
        instructionsScreenReDrawAll(canvas, data)
    elif data.mode == "webcamScreen": 
        webcamScreenReDrawAll(canvas,data)
    elif data.mode == "resultScreen": 
        resultScreenReDrawAll(canvas, data) 
    elif data.mode == "sleepScreen": 
        sleepScreenReDrawAll(canvas,data)  
    elif data.mode == "graphScreen": 
        graphScreenReDrawAll(canvas,data)
    elif data.mode == "aboutScreen": 
        aboutScreenReDrawAll(canvas,data)

###############################################################################
#animations 

def splashScreenMousedPressed(event,data): 
    if ((data.usernameBoxX <= event.x <= data.usernameBoxX + data.usernameBoxW) and 
        (data.usernameBoxY <= event.y <= data.usernameBoxY + data.usernameBoxL)): 
        data.usernameBox = True 
    if ((data.startButtonX <= event.x <= data.startButtonX+data.startButtonWidth) 
        and (data.startButtonY <= event.y <= data.startButtonY+data.startButtonLength)):
        if data.username != "": 
            data.mode = "instructionsScreen"
    if 1100 <= event.x <= 1200 and 600 <= event.y <= 700: 
        data.accuracy = analyzeAccuracy() * 100 
        data.mode = "aboutScreen"

def splashScreenKeyPressed(event, data): 
    if event.keysym in string.ascii_letters: 
        data.username += event.keysym
    if event.keysym == 'BackSpace': 
        data.username = data.username[:len(data.username)-1]

def splashScreenReDrawAll(canvas, data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick3")
    canvas.create_image(0,0, image = data.BaymaxImage, anchor = NW)
    canvas.create_rectangle(data.startButtonX,data.startButtonY,
    data.startButtonX+data.startButtonWidth,
    data.startButtonY+data.startButtonLength,fill = "white")
    canvas.create_text(1000,525,font="Times 20 bold italic", 
        text = "Let's Start!")
    canvas.create_text(900, 75, font = "Times 60 bold italic", 
        text= "Baymax", fill= "white")
    canvas.create_text(900, 150, font ='Times 40 bold italic', 
        text = "Your Personal Sleep Tracker", fill = "white")
    canvas.create_text(845, 325, text = 'username:', fill = 'white', font = 'Times 40 italic')
    canvas.create_rectangle(data.usernameBoxX, data.usernameBoxY, data.usernameBoxX+data.usernameBoxW, 
        data.usernameBoxY+data.usernameBoxL, fill ="white")
    canvas.create_text(1150, 650, text = '?', fill = 'white', font = 'Times 40')
    if data.usernameBox:
        canvas.create_rectangle(data.usernameBoxX, data.usernameBoxY, data.usernameBoxX+data.usernameBoxW, 
        data.usernameBoxY+data.usernameBoxL, fill ="yellow") 
        canvas.create_text(975, 325,text=data.username,font="Times 25",anchor=W)
def  aboutScreenMousePressed(event,data):
    if 1000 <= event.x <= 1100 and 100 <= event.y <= 1150: 
        data.mode = "splashScreen"

def aboutScreenReDrawAll(canvas,data):
    canvas.create_rectangle(0,0,1200,700, fill="firebrick3")
    canvas.create_rectangle(1000,100,1100,150, fill = "white")
    canvas.create_text(1050, 125, text ="Back", font = "Times 30")
    canvas.create_text(600, 350, text = str(data.accuracy), font = "Times 40")

def instructionsScreenMousedPressed(event, data):
    if ((data.continueButtonX <= event.x <= data.continueButtonX + data.continueButtonWidth) 
        and (data.continueButtonY <= event.y <= data.continueButtonY + data.continueButtonLength)):
        data.results = captureFrame(data)
        print(data.results)
        #if data.results == "error": 
            #data.error = True 
        data.mode = "webcamScreen"

def instructionsScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_text(600, 100, font = "Times 60 bold italic", 
        text = "Instructions", fill = "white" )
    canvas.create_text(150,200, font = "Times 25", 
        text = """1. Please make sure your face is in the webcam screen.""", 
        fill = "white", anchor = NW)
    canvas.create_text(150,250, font = "Times 25",
        text = """2. Please be in a well lit area.""", fill = "white", 
        anchor = NW)
    canvas.create_text(150, 300, 
        text = "3.Try to keep laptop or webcam at desk or face level.",
        font = "Times 25", fill ="white", anchor = NW)
    canvas.create_text(150,350, text ="4. Try to be in an empty setting.", font = "Times 25", fill = "white", anchor = NW)
    canvas.create_text(100, 150, 
        text = "To ensure best results, follow the following directions...", 
        fill = "white", anchor = NW, font = "Times 40")
    canvas.create_text(150,400, font = "Times 25", text = """5. Click on the camera icon when you are ready, and then press "P" to capture the frame.""", 
        fill = "white", anchor = NW )
    canvas.create_image(500,450,image = data.CameraImage, anchor = NW)

def webcamScreenMousePressed(event, data): 
    if ((data.nextScreenX <= event.x <= data.nextScreenX + data.nextScreenWidth) 
        and (data.nextScreenY <= event.y <= data.nextScreenY + data.nextScreenLength)):
            if data.results != None: 
                data.eye = data.results[0]
                data.mouth = data.results[1]
                data.state = data.results[2]
                data.mode = "resultScreen"

def webcamScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_image(0,0, image = data.baymax2Image, anchor = NW)
    canvas.create_rectangle(data.nextScreenX,data.nextScreenY, 
        data.nextScreenWidth+data.nextScreenX, data.nextScreenY+data.nextScreenLength, 
        fill = "white")
    canvas.create_text(900,515.5, text= """Let's Get Results!""", 
        fill = "black" , font = "Times 20")

def resultScreenMousePress(event,data): 
    if ((data.YesX <= event.x <= data.YesX+data.YesWidth) 
        and (data.YesY <= event.y <= data.YesY+data.YesLength)): 
        data.response = "yes"
        writeDatatoCSV(data.eye,data.mouth, data.state)
        data.mode = "sleepScreen"
    if ((data.NoX <= event.x <= data.NoX+data.NoWidth) 
        and (data.NoY <= event.y <= data.NoY+data.NoLength)): 
        if data.state == "sleepy": 
            data.state = "awake"
            data.response = "no"
            writeDatatoCSV(data.eye,data.mouth, data.state)
            data.mode = "sleepScreen"
        if data.state == "awake":
            data.state = "sleepy"
            data.response = "no"
            writeDatatoCSV(data.eye,data.mouth, data.state)
            data.mode = "sleepScreen"
    
def resultScreenReDrawAll(canvas, data):
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    state = str(data.state)
    canvas.create_text(600, 100, text = "It seems that you are...", 
        font = "Times 60", fill = "white")
    canvas.create_text(600, 470, text = state, font = "Times 40")
    canvas.create_rectangle(data.YesX, data.YesY, data.YesX+data.YesWidth, 
        data.YesY+data.YesLength, fill = "white")
    canvas.create_rectangle(data.NoX, data.NoY, data.NoX+data.NoWidth,
        data.NoY+data.NoLength, fill = "white")
    canvas.create_text(425, 625, text = "Yes", fill = "black", font = "Times 30")
    canvas.create_text(825, 625, text = "No", fill = "black", font = "Times 30")
    if data.state == "awake": 
        canvas.create_image(600,300, image = data.awakeEmojiImage)
    if data.state == "sleepy": 
        canvas.create_image(600, 300, image = data.sleepyEmojiImage)

def sleepScreenKeyPressed(event,data):
    if data.sleepBox:
        if event.keysym in string.digits: 
            data.sleepHours += event.keysym
        if event.keysym == 'BackSpace': 
            data.sleepHours = data.sleepHours[:len(data.username)-1]
    if data.dateBox: 
        if event.keysym in string.digits: 
            data.date += event.keysym
        if event.keysym == 'BackSpace': 
            data.date = data.date[:len(data.username)-1]

def sleepScreenMousePressed(event, data): 
    if ((data.sleepBoxX <= event.x <= data.sleepBoxX + data.sleepBoxWidth) and 
        (data.sleepBoxY <= event.y <= data.sleepBoxY + data.sleepBoxLength)): 
        data.sleepBox = True 
        data.dateBox = False 
    if ((data.dateX <= event.x <= data.dateX + data.dateWidth) and 
        (data.dateY <= event.y <= data.dateY +data.dateLength)): 
        data.dateBox = True 
        data.sleepBox = False 
    if data.sleepHours != None and data.date != None: 
        if ((data.EnterX <= event.x <= data.EnterX+data.EnterWidth) 
            and (data.EnterY <= event.y <= data.EnterY + data.EnterLength)):
                data.graph = plotGraph(data)
                data.graphImg = Image.open(data.graph)
                data.graphImg = data.graphImg.resize((600, 400), Image.ANTIALIAS)
                data.graphImage = ImageTk.PhotoImage(data.graphImg)
                data.mode = "graphScreen"


def sleepScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_text(600, 300, text ="How many hours of sleep did you get?", 
        font = "Times 50",fill = "white")
    canvas.create_text(600, 500, text = "What is today's date?", 
        font = "Times 50", fill = "white")
    canvas.create_rectangle(data.EnterX, data.EnterY, data.EnterX+data.EnterWidth, 
        data.EnterY+data.EnterLength, fill = "white")
    canvas.create_rectangle(data.dateX, data.dateY, 
        data.dateX + data.dateWidth, data.dateY+data.dateLength, fill = "white")
    canvas.create_rectangle(data.sleepBoxX,data.sleepBoxY, 
        data.sleepBoxX+data.sleepBoxWidth, data.sleepBoxY +data.sleepBoxLength, fill= "white")
    canvas.create_text(750, 525, text = "Enter", font = "times 20", fill = "black")
    if data.sleepBox or data.dateBox: 
        canvas.create_text(650, 375, text = data.sleepHours, font = "Times 20", 
            anchor = NW,fill = "black") 
        canvas.create_text(750, 475, text = data.date, font = "Times 20", 
            anchor = NW, fill = "black")

def graphScreenMousePressed(event,data): 
    if ((data.logOutX <= event.x <= data.logOutX + data.logOutWidth) 
        and (data.logOutY <= event.y <= data.logOutY + data.logOutLength)): 
        data.username = ""
        data.sleepHours = "" 
        data.date= "" 
        data.usernameBox = False 
        data.mode = "splashScreen"


def graphScreenReDrawAll(canvas, data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_image(600,400, image= data.graphImage)
    canvas.create_rectangle(data.logOutX,data.logOutY,
        data.logOutX+data.logOutWidth,data.logOutY+data.logOutLength,fill="white")
    canvas.create_text(1050, 125, text = "Log Out",fill = 'black',font = "Times 20")
    canvas.create_text(500, 50, text = "Your Sleep Data", font = "Times 40", fill = "white", anchor = NW)
    #open up 

###############################################################################
count = 0 
def captureFrame(data):
    cap = cv2.VideoCapture(0)
    while True: 
        ret, img = cap.read() 
        ret, img2 = cap.read()
        k = cv2.waitKey(1) & 0xff
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces: 
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_eyes_gray = gray[y:y+(4.9/10.0)*h, x:x+w]
            roi_eyes_color = img[y:y+(4.9/10.0)*h, x:x+w]
            roi_mouth_gray = gray[y+(6.5/10.0)*h:y+h, x:x+w]
            roi_mouth_color = img[y+(6.5/10.0)*h:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_eyes_gray,1.1,6)
            mouth = mouth_cascade.detectMultiScale(roi_mouth_gray,1.1,50)
            for ex,ey,eh,ew in eyes: 
                cv2.rectangle(roi_eyes_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
            for mx,my,mw,mh in mouth: 
                cv2.rectangle(roi_mouth_color, (mx,my), (mx+mw, my+mh), (0,0,255), 2)
        cv2.imshow('img', img)
        if k == 27: 
            cap.release()
            cv2.destroyAllWindows()  
            break 
        elif k == ord('p'): 
            cv2.imwrite("test.jpg", img2) 
            cap.release()
            cv2.destroyAllWindows() 
            break
    results = findROIs("test.jpg",data)
    print(results,"captureFrame")
    return results 

def findROIs(filename,data): 
    img = cv2.imread(filename)
    #this line of code stores the image read by openCV in the img variable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #stores the grayscale of img by converting the color to gray 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #detects faces using the face_cascade and the grayscale of the img 
    #the second parameter represents scaleFactor
    #the third parameter represents minimum neighbors 
    #if count >= 5: return "error"
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
        print(eyes)
        print(mouth)
        if len(eyes) == 2 and len(mouth) == 1: 
            ex, ey, ew, eh = eyes[0]
            eye1 = eyesClass(ex,ey,ew,eh)
            ex2, ey2, ew2, eh2 = eyes[1]
            eye2 = eyesClass(ex2, ey2, ew2,eh2)
            mx,my,mw,mh = mouth[0]
            mouth1 = mouthClass(mx,my,mw,mh)
        else: 
            print("error")
            captureFrame(data)
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
        print("Yay!! eyes")

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
        return (totalSpace - white)*.75

class mouthClass(facialFeatures): 
    def __init__(self, x,y,width,height):
        self.x = x 
        self.y = y 
        self.width = width 
        self.height = height
        self.croppedImage = self.cropImage()
        print("Yay!! mouth")

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
    eyeArea += eye1.calculateOpenEyePixels() + eye1.calculatePupilSize()
    eyeArea += eye2.calculateOpenEyePixels() + eye2.calculatePupilSize()
    totalEyeArea = eye1.calculateRoiPixels() + eye2.calculateRoiPixels()
    mouthArea = mouth1.calculateOpenMouthPixels()
    totalMouthArea = mouth1.calculateRoiPixels()
    return(eyeArea/(totalEyeArea*1.0), mouthArea/(1.0*totalMouthArea))

###############################################################################
#here starts the KNN Algorithm 

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
    classification = []
    for j in range(k):
        group = newDst[j][0][2].strip()
        classification.append(group)
    print(classification)
    #only has the 3-nearest nieghbors by classification 
    if classification.count("awake") > classification.count("sleepy"):
        return (x,y,"awake")
    elif classification.count("sleepy") > classification.count("awake"):
        return (x,y,"sleepy")

#writing data to a file
def writeDatatoCSV(x,y,classification): 
    with open("data.csv", "a") as datafile:
        dataFileWriter = csv.writer(datafile)
        dataFileWriter.writerow([x,y,classification])
    datafile.close()

def initializeData(filename):
    x,y,classification = findROIs(filename)
    writeDatatoCSV(x,y,classification)

def writeAccuracytoCSV(response): 
    with open("accuracy.csv","a") as accuracyFile: 
        accuracyFileWriter = csv.writer(accuracyFile)
        accuracyFileWriter.writerow([response])

def analyzeAccuracy():
    with open("accuracy.csv","rb") as analysisFile: 
        data = csv.reader(analysisFile)
        data = list(data)
        singleton = [] 
        for i in range(len(data)):
            singleton += data[i]
    success = singleton.count("yes")
    return (success * 1.0) / len(singleton)

def writeUsertoCSV(dictionary): 
    print("ayo")
    newDict = pickle.dump(dictionary, open( "save.p", "wb" ))

def userDictionary(): 
    currentDict = pickle.load( open( "save.p", "rb" ))
    return currentDict

def dictionaryModification(data): 
    userDict = userDictionary()
    if data.username in userDict: 
        currentValue = userDict[data.username]
        currentValue.append((data.date, data.sleepHours,data.state))
    else: 
        userDict[data.username] = []
        currentValue = userDict[data.username]
        currentValue.append((data.date,data.sleepHours,data.state))
    userDict[data.username] = currentValue
    print(userDict)
    writeUsertoCSV(userDict)
    return userDict


def plotGraph(data): 
    updatedDictionary = dictionaryModification(data)
    value = updatedDictionary[data.username]
    style.use('ggplot')
    sleepyXData = []
    sleepyYData = []
    awakeXData = []
    awakeYData = []
    for i in range(len(value)): 
        if value[i][2].strip() == "awake": 
            awakeXData.append(float(value[i][0]))
            awakeYData.append(float(value[i][1]))
        else: 
            sleepyXData.append(float(value[i][0]))
            sleepyYData.append(float(value[i][1]))

    hi = plt.scatter(awakeXData, awakeYData, color ='k', s=50)
    bye = plt.scatter(sleepyXData, sleepyYData, color='g',s=50)
    plt.legend((hi, bye),('awake', 'sleepy'),scatterpoints=1,fontsize=8)

    print(sleepyXData,sleepyYData, awakeXData, awakeYData)
    plt.title('Your Sleep Schedule')
    plt.ylabel('Number of Sleep Hours')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig("graph.png")
    return "graph.png"

###############################################################################
def run(width=750, height=750): #from Kosbie's Notes 
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    # create the root and the canvas
    root = Tk()
    init(data)
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

def main(): 
    #initializeData("data1.jpg")
    #initializeData("data2.jpg")
    #initializeData("data3.jpg")
    #initializeData("data4.jpg")
    #initializeData("data5.jpg")
    #initializeData("data6.jpg")
    #initializeData("data7.jpg")
    #initializeData("data8.jpg")
    #initializeData("data9.jpg")
    #initializeData("data10.jpg")
    #initializeData("data11.jpg")
    #initializeData("data12.jpg")
    #initializeData("data13.jpg")
    #initializeData("data14.jpg")
    #initializeData("data15.jpg")
    #initializeData("data16.jpg")
    #initializeData("data17.jpg")
    #initializeData("data18.jpg")
    #initializeData("data19.jpg")
    #initializeData("data20.jpg")
    run(1200,700)
main() 
