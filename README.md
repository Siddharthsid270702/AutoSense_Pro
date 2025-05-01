AutoSense Pro
AutoSense Pro is an AI and IoT-powered real-time driver safety monitoring system. It includes:
•	Drowsiness Detection using Eye Aspect Ratio (EAR)
•	Traffic Sign Detection via YOLOv5 object detection
•	Real-time Alerts using sound, LCD display, and Twilio SMS
•	Alcohol and Driver Cabin Temperature Monitoring through Arduino integration
________________________________________
Features
1. Drowsiness Detection
•	Uses a webcam and Dlib's 68 facial landmarks to monitor eye closure.
•	Plays an alarm and sends SMS if drowsiness is detected.
•	Displays alerts on a connected Arduino LCD.
2. Traffic Sign Detection
•	Uses a YOLOv5 model to detect traffic signs in real time.
•	Sends the detected sign label to Arduino for LCD display.
•	Avoids repeated notifications for the same sign.
3. Twilio SMS Integration
•	Sends alerts to a registered phone number using Twilio when driver drowsiness is detected or no face is found.
4. IoT Integration
•	Arduino receives alerts via serial communication for:
o	LCD display
o	Alcohol sensor
o	Engine temperature
________________________________________
🔌 Hardware Components Used
Component	Description
Arduino Uno	Microcontroller used for sensor interface
DHT11 Sensor	Measures temperature and humidity
MQ3 Sensor	Detects alcohol levels
LCD Display	Displays warning messages and traffic signs
________________________________________
Getting Started
Prerequisites
•	Python 3.8+
•	A working webcam
•	Arduino with sensors and LCD display
•	Pre-trained YOLOv5 model (best.pt)
•	Dlib shape predictor file (shape_predictor_68_face_landmarks.dat)
•	Twilio account and credentials
•	Audio device for alert (alarm.wav)
