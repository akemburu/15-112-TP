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
#splashScreen stuff 

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
        #captureFrame fucntion call 


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