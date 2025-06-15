from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
# Load the YOLO model (trained on traffic signs)
model = YOLO("./weights/last.pt")

# Class label list (index = class ID)
class_labels = [
    'bus_stop', 'do_not_enter', 'do_not_stop', 'do_not_turn_l', 'do_not_turn_r',
    'do_not_u_turn', 'enter_left_lane', 'green_light', 'left_right_lane', 'no_parking',
    'parking', 'ped_crossing', 'ped_zebra_cross', 'railway_crossing', 'red_light',
    'stop', 't_intersection_l', 'traffic_light', 'u_turn', 'warning', 'yellow_light'
]

# Function to process live video feed and detect traffic signs
def live_camera_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the camera.")
        return

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read from the camera.")
            return

        results = model.predict(source=frame, conf=0.5)
        annotated_frame = results[0].plot()

        detections = results[0].boxes
        for box in detections:
            cls = int(box.cls)
            conf = float(box.conf)
            if 0 <= cls < len(class_labels):
                label = class_labels[cls]
                print(f"Detected: {label} ({conf:.2f})")

                # Show popup on detection (only once per detection)
                #messagebox.showinfo("Traffic Sign Detected", f"Detected: {label} (Confidence: {conf:.2f})")
                break

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        annotated_frame = Image.fromarray(annotated_frame)
        annotated_frame = ImageTk.PhotoImage(annotated_frame)

        result_label.config(image=annotated_frame)
        result_label.image = annotated_frame

        root.after(10, process_frame)

    process_frame()

# Initialize the GUI
root = tk.Tk()
root.title("Real-time Traffic Sign Detection")
root.geometry("800x600")

title_label = tk.Label(root, text="AI Traffic Sign Detection", font=("Arial", 24, "bold"))
title_label.pack(pady=10)

button_camera = tk.Button(root, text="Start Live Camera Detection", font=("Arial", 14), bg="green", fg="white",
                          command=live_camera_detection)
button_camera.pack(pady=10)

result_label = tk.Label(root)
result_label.pack(pady=20)

root.mainloop()
