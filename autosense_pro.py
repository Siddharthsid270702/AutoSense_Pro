import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import serial
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pygame import mixer
import threading
from twilio.rest import Client  # Import Twilio Client

# Initialize the mixer for playing alert sounds
mixer.init()
mixer.music.load("alarm.wav")

# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC4ba59631ca4a644e4911e0d015465dbb'
TWILIO_AUTH_TOKEN = 'c5f6be25014d0bb59baa0657438f12ce'
TWILIO_PHONE_NUMBER = '+18156052509'
RECIPIENT_PHONE_NUMBER = '+919080052230'  # Your friend's phone number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)  # Initialize Twilio client

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Drowsiness detection parameters
thresh = 0.3  # EAR threshold for drowsiness
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
flag = 0
alerting = False
drowsiness_start_time = None

# Serial communication for temperature and alcohol
arduino = serial.Serial('COM4', 9600, timeout=1)

# Variables to keep track of the last sent traffic sign
last_traffic_sign = None

def display_on_lcd(message):
    arduino.write((message + '\n').encode())  # Send message to Arduino (LCD)

def send_sms_alert(message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

def detect_drowsiness():
    global flag, alerting, drowsiness_start_time
    cap_drowsiness = cv2.VideoCapture(0)  # Camera 1 for drowsiness detection
    
    if not cap_drowsiness.isOpened():
        print("Error: Could not open drowsiness camera.")
        return

    no_face_start_time = None
    no_face_thresh = 5  # Seconds to wait before alerting if no face is detected
    closed_eye_start_time = None  # Timer for when eyes are closed

    while True:
        ret, frame = cap_drowsiness.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        if len(subjects) == 0:
            # No face detected
            if no_face_start_time is None:
                no_face_start_time = time.time()

            if (time.time() - no_face_start_time) >= no_face_thresh:
                if not alerting:
                    mixer.music.play(-1)  # Play the alarm sound in a loop
                    alerting = True
                    display_on_lcd('Face Not Detected')

            # Reset drowsiness detection parameters since no face is detected
            flag = 0
            closed_eye_start_time = None

        else:
            # Reset no face timer
            no_face_start_time = None
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 510, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 510, 0), 1)

                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        if closed_eye_start_time is None:
                            closed_eye_start_time = time.time()  # Start the closed eyes timer
                        elif (time.time() - closed_eye_start_time >= 5):
                            if not alerting:
                                mixer.music.play(-1)  # Play the alarm sound in a loop
                                alerting = True
                                display_on_lcd('Drowsiness Detected')
                                send_sms_alert('Alert: Drowsiness Detected! Please take action.')  # Send SMS alert
                            cv2.putText(frame, "ALERT! Wake up!!!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "ALERT!", (10, 325),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    flag = 0  # Reset flag if eyes are open
                    closed_eye_start_time = None  # Reset closed eye timer
                    if alerting:
                        mixer.music.stop()
                        alerting = False

        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap_drowsiness.release()
    cv2.destroyAllWindows()

def detect_traffic_sign():
    global last_traffic_sign
    class Opt:
        weights = 'D:/Real-Time-Traffic-Sign-Detection/Model/Model/weights/best.pt'
        device = ''
        view_img = True
        save_txt = False
        save_conf = False
        classes = None
        agnostic_nms = False
        augment = False
        update = False
        project = 'runs/detect'
        name = 'exp'
        exist_ok = False
        conf_thres = 0.50
        iou_thres = 0.45
        img_size = 640

    opt = Opt()

    source = '1'  # Use camera number 0

    save_dir = Path(increment_path(Path("D:/demo/Real-Time-Traffic-Sign-Detection/Results") / opt.name, exist_ok=opt.exist_ok))  # increment run

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors for the bounding boxes.
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            h, w = im0.shape[:2]  # h,w
            save_path = str(save_dir / Path(p).name)  # TODO: add option to save or not
            txt_path = str(Path(save_path).with_suffix('.txt'))  # .txt
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    if opt.save_txt:  # Save to file
                        with open(txt_path, 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, conf))
                    if opt.view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        detected_sign = names[int(cls)]
                        
                        # Check if the detected sign is different from the last detected sign
                        if detected_sign != last_traffic_sign:
                            display_on_lcd(detected_sign)  # Send traffic sign to Arduino
                            last_traffic_sign = detected_sign  # Update the last detected sign

        # Display the results
        cv2.imshow('Traffic Sign Detection', im0)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    drowsiness_thread = threading.Thread(target=detect_drowsiness)
    traffic_sign_thread = threading.Thread(target=detect_traffic_sign)

    drowsiness_thread.start()
    traffic_sign_thread.start()

    drowsiness_thread.join()
    traffic_sign_thread.join()