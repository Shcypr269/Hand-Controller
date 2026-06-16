import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mouse
import numpy as np
import pyautogui
import os


def game():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )
    
    detector = vision.HandLandmarker.create_from_options(options)
    
    pressed = False
    cap = cv2.VideoCapture(0)
    cam_w, cam_h = 560, 480
    cap.set(3, cam_w)
    cap.set(4, cam_h)
    frameR = 100
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imageHeight, imageWidth, _ = img.shape
        
        # Convert to RGB for mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Get frame timestamp in milliseconds
        timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Get index finger tip (landmark 8)
                if len(hand_landmarks) > 8:
                    ind_x = int(hand_landmarks[8].x * imageWidth)
                    ind_y = int(hand_landmarks[8].y * imageHeight)
                    cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)
                    
                    conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, 1920)))
                    conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, 1080)))
                    mouse.move(conv_x, conv_y)
                    
                    # Count raised fingers
                    fingers = []
                    # Thumb
                    if hand_landmarks[4].x < hand_landmarks[3].x:
                        fingers.append(1)
                    # Index
                    if hand_landmarks[8].y < hand_landmarks[6].y:
                        fingers.append(1)
                    # Middle
                    if hand_landmarks[12].y < hand_landmarks[10].y:
                        fingers.append(1)
                    # Ring
                    if hand_landmarks[16].y < hand_landmarks[14].y:
                        fingers.append(1)
                    # Pinky
                    if hand_landmarks[20].y < hand_landmarks[18].y:
                        fingers.append(1)
                    
                    if fingers.count(1) == 2 and not pressed:
                        pyautogui.click()
                        pressed = True
                    elif fingers.count(1) != 2:
                        pressed = False

        cv2.imshow("Pointer", img)
        cv2.setWindowProperty("Pointer", cv2.WND_PROP_TOPMOST, 1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    detector.close()
