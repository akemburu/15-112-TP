from Tkinter import *
from PIL import Image, ImageTk 

#COMMENT
def init(data): 
    data.mode = "splashScreen"
    data.startButtonX = 900
    data.startButtonY = 500
    data.startButtonWidth = 200
    data.startButtonLength = 50
    data.continueButtonX = 500
    data.continueButtonY = 600
    data.continueButtonWidth = 200 
    data.continueButtonLength = 50
    data.Baymax = Image.open("Big_Hero_6_Baymax.gif")
    data.Baymax = data.Baymax.resize((700,700), Image.ANTIALIAS)
    data.BaymaxImage = ImageTk.PhotoImage(data.Baymax)
    data.Baymax2 = Image.open("baymax2.gif")
    data.Baymax2 = data.Baymax2.resize((400,250), Image.ANTIALIAS)
    data.Baymax2Image = ImageTk.PhotoImage(data.Baymax2)

def mousePressed(event, data): 
    if data.mode == "splashScreen": 
        splashScreenMousedPressed(event,data)
    if data.mode == "instructionsScreen": 
        instructionsScreenMousedPressed(event, data) 
    if data.mode == "webcamScreen": pass
    #if data.mode == "resultScreen": pass  



def keyPressed(event, data): 
    if data.mode == "splashScreen": pass 
    if data.mode == "instructionsScreen": pass 
    #if data.mode == "webcamScreen": 
    if data.mode == "resultScreen": pass 


def timerFired(data):
    pass

def redrawAll(canvas, data): 
    if data.mode == "splashScreen": 
        splashScreenReDrawAll(canvas, data)
    elif data.mode == "instructionsScreen": 
        instructionsScreenReDrawAll(canvas, data)
    #if data.mode == "webcamScreen": 
    elif data.mode == "resultScreen": 
        moodScreenReDrawAll(canvas, data) 


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
    canvas.create_text(800, 150, font = "Times 60 bold italic", 
        text= "Baymax", fill= "white")
    canvas.create_text(800, 230, font ='Times 40 bold italic', 
        text = "Your Personal Mood Detector", fill = "white")
    



def instructionsScreenMousedPressed(event, data):
    if ((data.continueButtonX <= event.x <= data.continueButtonX + data.continueButtonWidth) 
        and (data.continueButtonY <= event.y <= data.continueButtonY + data.continueButtonLength)):
        data.mode = "webcamScreen"

def instructionsScreenReDrawAll(canvas,data): 
    canvas.create_rectangle(0,0,1200,700, fill="firebrick1")
    canvas.create_rectangle(data.continueButtonX,data.continueButtonY,
        data.continueButtonX+data.continueButtonWidth,
        data.continueButtonY+data.continueButtonLength,fill = "white")
    canvas.create_text(600, 100, font = "Times 60 bold italic", 
        text = "Instructions", fill = "white" )
    canvas.create_text(100,300, font = "Times 40", 
        text = """1. Please make sure your face is in the webcam screen.""", 
        fill = "white", anchor = NW)
    canvas.create_text(100,350, font = "Times 40",
        text = """2. Please be in a well lit area.""", fill = "white", 
        anchor = NW)
    canvas.create_text(100, 400, 
        text = "3.Try to keep laptop or webcam at desk or face level.",
        font = "Times 40", fill ="white", anchor = NW)
    canvas.create_text(100, 200, 
        text = "To ensure best results, follow the following directions...", 
        fill = "white", anchor = NW, font = "Times 40")
    canvas.create_text(600, 625, text= "Continue", font = "Times 20 bold italic")
    canvas.create_image(0,450,image = data.Baymax2Image, anchor = NW)




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