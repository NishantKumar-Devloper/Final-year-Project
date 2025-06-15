import tkinter as tk
from tkinter import LEFT
from PIL import Image, ImageTk
from tkinter import messagebox as ms
from subprocess import call
import os

root = tk.Tk()
root.configure(background="brown")

# Get screen width and height
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("System")

# ----- Background Image -----
try:
    image2 = Image.open('./img1.jpg')
    image2 = image2.resize((w, h), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(image2)
    background_label = tk.Label(root, image=background_image)
    background_label.image = background_image
    background_label.place(x=0, y=0)
except Exception as e:
    ms.showerror("Error", f"Background image load failed:\n{e}")

# ----- Title -----
label_l1 = tk.Label(root, text="Road Sign Detection", font=("Times New Roman", 35, 'bold'),
                    background="Black", fg="white", width=30, height=1)
label_l1.place(x=300, y=20)

# ----- Button Functions -----
def log():
    try:
        call(["python3", "./testing_file.py"])
    except Exception as e:
        ms.showerror("Error", f"Failed to launch detection:\n{e}")

def text():
    try:
        call(["python3", "./text_extraction.py"])
    except Exception as e:
        ms.showerror("Error", f"Failed to launch text extraction:\n{e}")

def window():
    root.destroy()

# ----- Buttons -----
tk.Button(root, text="Detect Road Sign", command=log, width=20, height=1,
          font=('times', 20, 'bold'), bg="black", fg="white").place(x=100, y=200)

tk.Button(root, text="Text Extraction", command=text, width=20, height=1,
          font=('times', 20, 'bold'), bg="black", fg="white").place(x=100, y=300)

tk.Button(root, text="Exit", command=window, width=20, height=1,
          font=('times', 20, 'bold'), bg="#FF0000", fg="white").place(x=100, y=400)

root.mainloop()
