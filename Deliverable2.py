import numpy as np 
import cv2 
import math 
from Tkinter import *
import csv
from PIL import Image, ImageTk 


#each eye and mouth will be stores as objects from eye class 
#this way they can be accessed and used from inner methods  
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
#these cascades are data from an online which provide information to detect 
#a face, eyes, and mouth. 
#from: http://alereimondo.no-ip.org/OpenCV/34
from Tkinter import *
from PIL import Image, ImageTk 

def init(data): 
    data.mode = "splashScreen"
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
    data.nextScreenY = 500
    data.nextScreenWidth = 200
    data.nextScreenLength = 50 
    #data.Baymax2 = Image.open("baymax2.gif")
    #data.Baymax2 = data.Baymax2.resize((400,250), Image.ANTIALIAS)
    #data.Baymax2Image = ImageTk.PhotoImage(data.Baymax2)
    data.Camera = Image.open("camera.gif")
    data.Camera = data.Camera.resize((200,200), Image.ANTIALIAS)
    data.CameraImage = ImageTk.PhotoImage(data.Camera)    

def mousePressed(event, data): 
    if data.mode == "splashScreen": 
        splashScreenMousedPressed(event,data)
    if data.mode == "instructionsScreen": 
        instructionsScreenMousedPressed(event, data) 
    if data.mode == "webcamScreen": webcamScreenMousePressed(event,data)
    #if data.mode == "resultScreen": pass  


def keyPressed(event, data): 
    if data.mode == "splashScreen": pass 
    if data.mode == "instructionsScreen": pass 
    if data.mode == "webcamScreen": pass
    if data.mode == "resultScreen": pass 


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


###############################################################################
#animations 

def splashScreenMousedPressed(event,data): 
    if ((data.startButtonX <= event.x <= data.startButtonX + data.startButtonWidth) 
        and (data.startButtonY <= event.y <= data.startButtonY + data.startButtonLength)):
        data.mode = "instructionsScreen"

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
    


def instructionsScreenMousedPressed(event, data):
    if ((data.continueButtonX <= event.x <= data.continueButtonX + data.continueButtonWidth) 
        and (data.continueButtonY <= event.y <= data.continueButtonY + data.continueButtonLength)):
        data.mode = "webcamScreen"
        captureFrame()

def instructionsScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_text(600, 100, font = "Times 60 bold italic", 
        text = "Instructions", fill = "white" )
    canvas.create_text(100,200, font = "Times 40", 
        text = """1. Please make sure your face is in the webcam screen.""", 
        fill = "white", anchor = NW)
    canvas.create_text(100,250, font = "Times 40",
        text = """2. Please be in a well lit area.""", fill = "white", 
        anchor = NW)
    canvas.create_text(100, 300, 
        text = "3.Try to keep laptop or webcam at desk or face level.",
        font = "Times 40", fill ="white", anchor = NW)
    canvas.create_text(100, 150, 
        text = "To ensure best results, follow the following directions...", 
        fill = "white", anchor = NW, font = "Times 40")
    canvas.create_text(100,350, font = "Times 40", text = """4. Click on the camera icon when you are ready, 
        and then press "P" to capture the frame.""", 
        fill = "white", anchor = NW )
    canvas.create_image(500,450,image = data.CameraImage, anchor = NW)

def webcamScreenMousePressed(event, data): 
    if ((data.nextScreenX <= event.x <= data.nextScreenX + data.nextScreenWidth) 
        and (data.nextScreenY <= event.y <= data.nextScreenY + data.nextScreenLength)):
        data.mode = "resultScreen"

def webcamScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_text(420,325, text = "Picture Time!", font = "Times 80", 
        anchor = NW, fill = "white")
    canvas.create_rectangle(data.nextScreenX,data.nextScreenY, 
        data.nextScreenWidth+data.nextScreenX, data.nextScreenY+data.nextScreenLength, fill = "white")
    canvas.create_text(900,515.5, text= """Let's Get Results!""", fill = "black" , font = "Times 20")

def resultScreenReDrawAll(canvas, data):
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_text(600, 100, text = "It seems that you are...", font = "Times 60", fill = "white")

###############################################################################
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
    print(faces)
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
        finalAnswer = neighbors(eye1,eye2,mouth1)


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
    print(eyeArea/(totalEyeArea*1.0), mouthArea/(1.0*totalMouthArea))
    return (eyeArea/totalEyeArea, mouthArea/totalMouthArea)

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
    #check to make sure that every number is a decimal 
    return dataSet
 
def calculateDistance(x,y,x1,y1):
    distance = (x-x1)**2 + (y-y1)**2
    return math.sqrt(distance)
    
def getKey(item):
    return item[1]

def neighbors(eye1,eye2,mouth1,k=3,filename = "data.csv"):
    #where x defines size of eyes
    #where y defines size of mouth 
    x, y = calculateAreas(eye1,eye2,mouth1) 
    dataSet = makeDataSet(filename)
    allDists = [] 
    for i in range(len(dataSet)):
        x1 = dataSet[i][0]
        y1 = dataSet[i][1]
        distance = calculateDistance(x,y,x1,y1)
        allDists.append((dataSet[i], distance))
    newDst = sorted(allDists, key = getKey)
    neighbors = []
    for j in range(k):
        neighbors.append(newDst[j][0])
    print(neighbors)
    #return neighbors
    classification(neighbors)

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


#writing data to a file 
#def writeDatatoCSV(x,y,classification): 
    #with open("data.csv", "w") as datafile:
        #dataFileWriter = csv.writer(datafile)
        #dataFileWriter.writerow([x,y,classification])

###############################################################################
def run(width=750, height=750):
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

run(1200,700)

