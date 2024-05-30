from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from app.verify import authentication
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from datetime import datetime
import cv2
from time import time
import mediapipe as mp
from .process import *



# Create your views here.
def index(request):
    # return HttpResponse("This is Home page")    
    return render(request, "index.html")

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html")

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html")


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname': request.user.first_name,
        }
    if request.method == "POST":
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

    return render(request, "dashboard.html",context)