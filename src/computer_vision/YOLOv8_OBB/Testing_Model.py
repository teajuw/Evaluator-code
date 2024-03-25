import os
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

# Load the model
model = YOLO('best.pt')  # Path to your model file

file_name = "frame_19.jpg"
# results = model(file_name)

# print(results[0])

import supervision as sv
import cv2
import mediapipe as mp

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
frame = cv2.flip(frame, 1)
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

results = model(frame)

detections = sv.Detections.from_ultralytics(results[0])

oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=cv2.imread(file_name),
    detections=detections
)

sv.plot_image(image=annotated_frame, size=(16, 16))