from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import time
import cv2
import pygame
from gtts import gTTS
import os
import math
import tkinter as tk
from PIL import Image, ImageTk  # Importing Image and ImageTk from PIL
import tempfile


def pmusic(file):
    pygame.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(1000)
    pygame.mixer.quit()

def stopmusic():
    pygame.mixer.music.stop()

def getmixerargs():
    pygame.mixer.init()
    freq, size, chan = pygame.mixer.get_init()
    return freq, size, chan

def initMixer():
    BUFFER = 4096  # audio buffer size, number of samples since pygame 1.8.
    FREQ, SIZE, CHAN = getmixerargs()
    pygame.mixer.init(FREQ, SIZE, CHAN, BUFFER)





# Load YOLO model
model = YOLO(".\\weights\\best.pt")

# Initialize Tkinter
root = tk.Tk()
root.title("YOLO Object Detection")

# Open camera
cap = cv2.VideoCapture(0)

title_label = tk.Label(root, text="AI Traffic Sign Detection", font=("Arial", 24, "bold"))
title_label.pack(pady=10)

# Create a Tkinter label to display video feed
label = tk.Label(root)
label.pack()

# Variable to store detected classes
detected_text = tk.StringVar()
detected_label = tk.Label(root, textvariable=detected_text, font=("Arial", 20,'bold'))
detected_label.pack()

def update_frame():
    """ Continuously updates the camera feed in Tkinter window. """
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
    root.after(10, update_frame)  # Refresh frame every 10ms

def capture_and_detect():
    """ Captures an image, runs YOLO detection, and updates the Tkinter UI. """
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        return

    # Run YOLO detection
    results = model(frame)
    
    # Process results
    detected_classes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)  # Class ID
            conf = float(box.conf)  # Confidence score
            names =  [
                'bus stop', 'do not enter', 'do not stop', 'do not turn left', 'do not turn right',
                'do not u turn', 'enter left lane', 'green light', 'left right lane', 'no parking',
                'parking', 'ped crossing', 'ped zebra cross', 'railway crossing', 'red light',
                'stop', 't intersection left', 'traffic light', 'u turn', 'warning', 'yellow light'
            ]
 
            detected_classes.append(f"Class: {names[cls]}, Confidence: {conf:.2f}")
            print("Class ID--------------------->"+str(cls))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                        tts = gTTS(text=str(names[cls]), lang='en', slow=False)
                        tts.save(temp_audio_file.name)
                        temp_audio_name = temp_audio_file.name  # Store the file name

            pmusic(temp_audio_name)
            os.remove(temp_audio_name)  # Clean up the temp file after playback'''
               
        # Annotate frame with YOLO detections
        annotated_frame = result.plot()

        # Convert to Tkinter format
        img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Update the label with the new image
        label.config(image=img)
        label.image = img  

    # Update detected class text
    detected_text.set("\n".join(detected_classes) if detected_classes else "No objects detected.")
    
        

# Bind 'C' key to capture function
def key_press(event):
    if event.char.lower() == 'c':
        capture_and_detect()

# Bind keypress event to Tkinter window
root.bind("<KeyPress>", key_press)

# Start updating camera feed
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release camera after closing
cap.release()
cv2.destroyAllWindows()
