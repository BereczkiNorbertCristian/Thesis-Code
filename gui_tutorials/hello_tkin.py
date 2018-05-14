
import tkinter as tk
from PIL import ImageTk,Image

root = tk.Tk()

img_file = Image.open("image_1.jpg")
img = ImageTk.PhotoImage(img_file)

txt =  "Hello world by Norbert HHHHHHHHello world by Norbertello world by Norbertello world by Norbertello world by Norbertello world by Norbertello world by Norbertello world by Norbertello world by Norbert"


w1 = tk.Label(root,image=img).pack(side="right")
w2 = tk.Label(root,justify = tk.LEFT,padx = 10,text=txt).pack(side="left")

root.mainloop()

