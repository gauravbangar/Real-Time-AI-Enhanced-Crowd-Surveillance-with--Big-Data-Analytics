import cv2
import numpy as np
import torch
from argparse import ArgumentParser
import clip
from packaging import version
from PIL import Image
import pygame
import threading

# Initialize flags and variables for fire detection
Fire_Reported = False
Sound_Played = False
Alarm_Status = False

# Initialize flags and variables for falling person detection
Fall_Reported = False

# Initialize flags and variables for person counting
Crowded_Area = False

# Initialize flags and variables for email notification
Email_Status = False

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load('ViT-B/32', device)
preprocess = transform

# Define text targets for CLIP model
targets = [
    "a man lying down on the floor",
    "a man standing up",
    "no people"
]
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in targets]).to(device)
text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Load YOLO model for fire detection
net_fire = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")

# Load YOLO model for person counting and falling person detection
net_person = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")

# Load class names for YOLO model
with open("models/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define function for playing alarm sound
def play_alarm_sound_function():
    global Sound_Played
    if not Sound_Played:
        pygame.mixer.init()
        pygame.mixer.music.load('sound/alarm-sound.mp3')
        pygame.mixer.music.play()
        Sound_Played = True

# Define function for drawing text box on frame
def draw_text_box(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, font_thickness=2, 
                  text_color=(0, 255, 0), text_color_bg=(0, 0, 0), margin=3):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x - margin, y - margin), (x + text_w + margin, y + text_h + margin), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size

# Define function for inference with CLIP model
def infer(src, text_features) -> (bool, torch.tensor):
    image_input = preprocess(src).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(targets))
        results = list(zip(values, indices))

        # Print the result
        for value, index in results:
            if index == 0 and value > 0.1:
                return True, results
    
    return False, results

# Function for fire detection
def detect_fire(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net_fire.setInput(blob)
    output_layer_names = net_fire.getUnconnectedOutLayersNames()
    layer_outputs = net_fire.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # Minimum confidence threshold for detection
    conf_threshold = 0.5

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and class_id == 1:  # 1 corresponds to 'fire' class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    if len(indices) > 0:
        return True
    else:
        return False

# Function for detecting falling person
def detect_fall(frame):
    # Add your falling person detection logic here
    # Return True if a fall is detected, False otherwise
    return False

# Function to convert numpy array to PIL image
def numpy_to_pil(image):
    return Image.fromarray(np.uint8(image))