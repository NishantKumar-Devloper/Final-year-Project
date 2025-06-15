import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
from google.cloud import vision
import pyttsx3

# Set your Google Cloud Vision API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './electricity-374907-14faa175fc81.json'

# Initialize Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up the Tkinter GUI
root = tk.Tk()
root.title("Live Camera - Press 'q' to Capture and Extract Text")

# Image label to show live/captured frame
image_label = tk.Label(root)
image_label.pack()

# Text label to display extracted text
text_label = tk.Label(root, text="", font=("Helvetica", 12), wraplength=600, justify="left")
text_label.pack(pady=10)

# Update live video feed in the GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
    root.after(10, update_frame)

# Capture and extract text when 'q' is pressed
def capture_and_extract(event=None):
    ret, frame = cap.read()
    if ret:
        # Save captured image
        img_path = 'captured_image.png'
        cv2.imwrite(img_path, frame)

        # Show the captured image
        img = Image.open(img_path).resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Load image for Google Vision
        with open(img_path, 'rb') as f:
            content = f.read()
        image = vision.Image(content=content)

        # Extract text using Google Vision API
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            extracted_text = texts[0].description.strip()
            text_label.config(text="Extracted Text:\n" + extracted_text)
            speak_text(extracted_text)
        else:
            text_label.config(text="No text detected.")
            speak_text("No text detected.")

# Bind 'q' key press to capture and extract
root.bind('<KeyPress>', lambda event: capture_and_extract() if event.char.lower() == 'q' else None)

# Graceful shutdown
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start live camera feed
update_frame()

# Start GUI loop
root.mainloop()
