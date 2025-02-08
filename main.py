import cv2
import time
import base64
import threading
import numpy as np
import os
from datetime import datetime
from picamera2 import Picamera2
from inference_sdk import InferenceHTTPClient
from libcamera import Transform
import RPi.GPIO as GPIO
import asyncio
from pyrebase import pyrebase

class InferenceCamera:
    def __init__(self, api_url, api_key, model_id, save_dir="captured_images"):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.picam2 = Picamera2()
        self.recycled = 0
        self.trashed = 0
        config = {
            "apiKey": "AIzaSyB7pTnsNqgx0ubEn5AfEBHYner0Onzu71o",
            "authDomain": "ecosort-c044a.firebaseapp.com",
            "databaseURL": "https://ecosort-c044a-default-rtdb.firebaseio.com/",
            "storageBucket":"ecosort-c044a.appspot.com",
        }
        firebase = pyrebase.initialize_app(config) 
        self.db = firebase.database()
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Camera configuration
        sizex, sizey = 1280, 720
        mode = self.picam2.sensor_modes[1]  # Selecting a high-quality sensor mode
        config = self.picam2.create_preview_configuration(
            sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']},
            main={"format": "RGB888", "size": (sizex, sizey)},
            lores={"size": (sizex, sizey)},
            transform=Transform(hflip=True, vflip=True)
        )

        self.picam2.configure(config)
        self.picam2.set_controls({"FrameRate": 40})
        self.picam2.start()

        # OpenCV window
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Feed", sizex, sizey)

        print("Camera started... Press SPACE to capture image, 'q' to quit.")

        # Thread control
        self.last_infer_time = 0
        self.inference_thread = None
        self.lock = threading.Lock()

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(32,GPIO.OUT)
        GPIO.setup(33,GPIO.OUT)
        self.back = GPIO.PWM(32, 50)
        self.front = GPIO.PWM(33, 50)
        self.prev = None
        self.counter = 0
        self.front.start(0)
        self.back.start(0)
    
    def center(self):
        self.back.ChangeDutyCycle(7.4)
        self.front.ChangeDutyCycle(8.6) 
        # await asyncio.sleep(0.2)
    def right(self):
        self.front.start(0)
        self.back.start(0)
        self.front.ChangeDutyCycle(11)
        self.back.ChangeDutyCycle(5)
        time.sleep(1)
        self.center()
        time.sleep(2)
        self.rest()

    def left(self):
        self.front.start(0)
        self.back.start(0)
        self.front.ChangeDutyCycle(6)
        self.back.ChangeDutyCycle(10.5)
        time.sleep(1)
        self.center()
        time.sleep(2)
        self.rest()

    def rest(self):
        self.front.start(0)
        self.back.start(0)

    def encode_image(self, frame):
        """Convert image to base64 encoding for inference API."""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    
    def infer_image(self, frame):
        """Perform inference using the API while ignoring 'cardboard'."""
        encoded_image = self.encode_image(frame)
        result = self.client.infer(encoded_image, model_id=self.model_id)
        
        predictions = result.get("predictions", [])

        # Remove 'cardboard' from predictions
        filtered_predictions = [p for p in predictions if p["class"].lower() != "cardboard"]

        if filtered_predictions:
            best_prediction = max(filtered_predictions, key=lambda x: x["confidence"])
            
            top_class = best_prediction["class"]
            if self.prev == None:
                self.prev = top_class
            top_confidence = best_prediction["confidence"]

            if top_confidence > 0.85:
                print(top_class)
                print(self.prev)
                print(self.counter)
                self.counter +=1
                if(self.prev != top_class and self.counter != 0):
                    print("mismatch")
                    self.counter = 0
                if(self.counter>=2):
                    if top_class == "recycle":
                        self.recycled+=1
                        self.right()
                    if top_class == "trash":
                        self.trashed+=1
                        self.left()
                    data = {
                        "recycling": self.recycled,
                        "trash": self.trashed
                    }
                    self.db.child("Status").push(data)
                    self.db.update(data)
                    self.counter = 0
                self.prev = top_class
            else:
                self.rest()
                self.counter = 0

                # print(f"Top predicted class: {top_class} (Confidence: {top_confidence:.2f})")

    def inference_task(self, frame):
        """Run inference in a separate thread."""
        with self.lock:
            self.infer_image(frame)

    def save_image(self, frame):
        """Save the captured frame as an image file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")

    def capture_and_infer(self, stagger):
        """Continuously capture frames, run inference, and allow image saving."""
        prev_frame_time = 0

        try:
            while True:
                # Capture frame from camera
                frame = self.picam2.capture_array("main")

                # Calculate FPS
                new_frame_time = time.time()
                fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time

                # Display FPS on frame
                cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

                # Display the frame
                # cv2.imshow("Camera Feed", frame)

                # Run inference every `stagger` seconds
                if time.time() - self.last_infer_time >= stagger:
                    self.last_infer_time = time.time()
                    if self.inference_thread is None or not self.inference_thread.is_alive():
                        self.inference_thread = threading.Thread(target=self.inference_task, args=(frame,))
                        self.inference_thread.start()

                # Capture image on SPACE key
                key = cv2.waitKey(1) & 0xFF

                # Quit on pressing 'q'
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Stopping camera...")
        
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()

# Usage Example
if __name__ == "__main__":
    api_url = "https://detect.roboflow.com"
    api_key = "aP6xXzQNC5cliLsDqI4u"
    model_id = "ecosortmax/2"
    
    camera = InferenceCamera(api_url, api_key, model_id)
    camera.capture_and_infer(0)  
