from PIL import Image, ImageTk 
import Tkinter as tk 








root = tk.Tk()
canvas = tk.Canvas(root, width=1000, height=800)
canvas.pack()
img = Image.open("Big_Hero_6_Baymax.gif")
tk_img = ImageTk.PhotoImage(img)
canvas.create_image(500, 350, image=tk_img)
root.mainloop()