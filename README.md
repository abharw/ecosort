# EcoSort 

This project involves capturing images using a Raspberry Pi camera, running object detection inference on the captured frames, and controlling a servo motor based on the detected objects. The system categorizes the objects into "recyclable" and "trash" and records the data to Firebase.

## Features
- **Camera Capture**: Captures images using a Raspberry Pi camera.
- **Object Detection**: Runs inference on each captured frame using a machine learning model.
- **Servo Motor Control**: Controls a servo motor to sort recyclable and trash objects.
- **Firebase Integration**: Updates Firebase with the number of recyclable and trash items detected.
- **Real-time FPS Display**: Shows the frame rate (FPS) of the camera feed.

## Requirements

### Hardware
- Raspberry Pi 4 (or similar)
- Raspberry Pi Camera Module
- 2 Servo motors (for sorting recyclable and trash items)
- GPIO pins for servo motor control
- Raspberry Pi power supply

### Software
- Raspberry Pi OS (Raspbian)
- Python 3.7 or higher
- Dependencies:
  - `picamera2`: For interfacing with the Raspberry Pi camera
  - `cv2` (OpenCV): For capturing frames and displaying the camera feed
  - `numpy`: For array manipulation
  - `pyrebase`: For Firebase integration
  - `inference_sdk`: For object detection inference
  - `RPi.GPIO`: For GPIO control of the servo motors
  - `asyncio`: For asynchronous tasks
  - `time`: For time-based operations
  - `base64`: For encoding images for inference
  
### The Website: 
[EcoSort](https://ecosortplus.netlify.app)
