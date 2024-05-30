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
net_fire = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load YOLO model for person counting and falling person detection
net_person = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names for YOLO model
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define function for playing alarm sound
def play_alarm_sound_function():
    global Sound_Played
    if not Sound_Played:
        pygame.mixer.init()
        pygame.mixer.music.load('alarm-sound.mp3')
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

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Fire detection
    fire_detected = detect_fire(frame)

    # Perform fire detection using color thresholding
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]  # Lower bound for red color in HSV
    upper = [35, 255, 255]  # Upper bound for red color in HSV
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    # Count the number of red pixels in the mask
    no_red = cv2.countNonZero(mask)

    # If a significant amount of red pixels is detected, consider it as fire
    if no_red > 15000:
        Fire_Reported = True
    else:
        Fire_Reported = False
        Sound_Played = False  # Reset flag when fire is not reported

    cv2.imshow("output", output)

    # Fire alarm logic
    if Fire_Reported and not Alarm_Status:
        threading.Thread(target=play_alarm_sound_function).start()
        Alarm_Status = True

    if not Fire_Reported:
        Alarm_Status = False


    # Perform YOLO object detection for person counting and falling person detection
    height, width = frame.shape[:2]
    blob_person = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net_person.setInput(blob_person)
    layer_names_person = net_person.getUnconnectedOutLayersNames()
    outputs_person = net_person.forward(layer_names_person)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes_person = []
    confidences_person = []
    class_ids_person = []

    # Minimum confidence threshold for detection
    conf_threshold_person = 0.5

    # Loop over each output layer
    for output_person in outputs_person:
        for detection_person in output_person:
            scores_person = detection_person[5:]
            class_id_person = np.argmax(scores_person)
            confidence_person = scores_person[class_id_person]

            if confidence_person > conf_threshold_person:
                if classes[class_id_person] == "person":  # Check if the detected object is a person
                    box_person = detection_person[0:4] * np.array([width, height, width, height])
                    (center_x_person, center_y_person, box_width_person, box_height_person) = box_person.astype("int")
                    x_person = int(center_x_person - (box_width_person / 2))
                    y_person = int(center_y_person - (box_height_person / 2))

                    boxes_person.append([x_person, y_person, int(box_width_person), int(box_height_person)])
                    confidences_person.append(float(confidence_person))
                    class_ids_person.append(class_id_person)

    # Apply non-maximum suppression to remove overlapping boxes
    nms_threshold_person = 0.3
    indices_person = cv2.dnn.NMSBoxes(boxes_person, confidences_person, conf_threshold_person, nms_threshold_person)

    # Person counting
    person_count = len(indices_person) if len(indices_person) > 0 else 0

    # Display the person count in the left top corner
    cv2.rectangle(frame, (0, 0), (700, 50), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, f"Persons: {person_count}", (450, 23), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)  # Green text

    if person_count > 3:
        cv2.putText(frame, f"Crowded Area!!", (450, 43), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)  # Red text

    # Loop over the indices for person counting and falling person detection
    if len(indices_person) > 0:
        for i_person in indices_person.flatten():
            box_person = boxes_person[i_person]
            x_person, y_person, w_person, h_person = box_person
            label_person = str(classes[class_ids_person[i_person]])
            confidence_person = confidences_person[i_person]
            color_person = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x_person, y_person), (x_person + w_person, y_person + h_person), color_person, 2)
            text_person = f"{label_person}: {confidence_person:.2f}"
            cv2.putText(frame, text_person, (x_person, y_person - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_person, 2)

    # Perform CLIP inference
    warn, pred = infer(numpy_to_pil(frame), text_features)
    for i, (value, index) in enumerate(pred):
        text_clip = f"{targets[index]}: {100 * value.item():.2f}%"
        (w_clip, h_clip), b_clip = cv2.getTextSize(text_clip, cv2.FONT_HERSHEY_DUPLEX, 0.5, thickness=None)
        cv2.rectangle(frame, (0, i * 16 + 16 - b_clip // 2), (0 + w_clip, i * 16 + 16 + h_clip + b_clip // 2), color=(0, 0, 0),
                      thickness=cv2.FILLED)
        cv2.putText(frame, text_clip, (0, i * 16 + 16 + h_clip), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                    lineType=cv2.LINE_AA)
        
    if warn:
        draw_text_box(frame, "FALLING PERSON DETECTED", pos=(0, 450), text_color=(0, 0, 0), text_color_bg=(64, 64, 255))

    # Display the frame with object detection and CLIP inference
    cv2.imshow("Combined Detection", frame)

    # Fire alarm logic
    if fire_detected and not Sound_Played:
        threading.Thread(target=play_alarm_sound_function).start()
        Sound_Played = True

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
