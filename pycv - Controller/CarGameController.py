import math
import keyinput
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import numpy as np


def draw_steering_wheel(image, center, radius, angle, hand_positions, action="straight"):
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius)
    
    # Colors based on action
    colors = {
        "straight": ((100, 255, 100), (0, 200, 50)),      # Green
        "left": ((100, 150, 255), (0, 100, 200)),         # Blue
        "right": ((100, 150, 255), (0, 100, 200)),        # Blue
        "drift": ((50, 50, 255), (0, 0, 200)),            # Red
        "nitro": ((200, 100, 255), (150, 0, 200)),        # Purple
        "back": ((100, 100, 100), (50, 50, 50))           # Gray
    }
    
    primary_color, secondary_color = colors.get(action, colors["straight"])
    
    # Draw outer ring
    for i in range(8, 0, -1):
        alpha = i / 10
        color = tuple(int(c * alpha + 50 * (1 - alpha)) for c in primary_color)
        cv2.circle(image, (cx, cy), radius + i, color, 2)
    
    # Draw main wheel circle
    cv2.circle(image, (cx, cy), radius, primary_color, 12)
    cv2.circle(image, (cx, cy), radius - 8, secondary_color, 3)
    
    # Draw wheel spokes based on rotation angle
    spoke_angle = angle * math.pi / 180
    
    # Calculate spoke endpoints
    spoke_length = radius - 15
    for i in range(3):
        angle_rad = spoke_angle + (i * 2 * math.pi / 3)
        x1 = int(cx + 20 * math.cos(angle_rad))
        y1 = int(cy + 20 * math.sin(angle_rad))
        x2 = int(cx + spoke_length * math.cos(angle_rad))
        y2 = int(cy + spoke_length * math.sin(angle_rad))
        cv2.line(image, (x1, y1), (x2, y2), primary_color, 8)
        cv2.line(image, (x1, y1), (x2, y2), secondary_color, 4)
    
    # Draw center hub
    cv2.circle(image, (cx, cy), 25, primary_color, -1)
    cv2.circle(image, (cx, cy), 20, secondary_color, -1)
    
    # Draw direction indicator (arrow on top of wheel)
    arrow_angle = spoke_angle - math.pi / 2
    arrow_x = int(cx + (radius - 5) * math.cos(arrow_angle))
    arrow_y = int(cy + (radius - 5) * math.sin(arrow_angle))
    
    # Draw arrow triangle
    arrow_size = 12
    pts = np.array([
        [arrow_x, arrow_y],
        [arrow_x - arrow_size // 2, arrow_y + arrow_size],
        [arrow_x + arrow_size // 2, arrow_y + arrow_size]
    ], np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 255))
    cv2.polylines(image, [pts], True, (0, 0, 0), 2)
    
    # Draw hand position indicators
    for i, (hx, hy) in enumerate(hand_positions):
        color = (0, 255, 255) if i == 0 else (255, 0, 255)
        cv2.circle(image, (int(hx), int(hy)), 15, color, -1)
        cv2.circle(image, (int(hx), int(hy)), 15, (0, 0, 0), 2)
        cv2.circle(image, (int(hx), int(hy)), 8, (255, 255, 255), -1)
    
    # Draw action text
    action_texts = {
        "straight": "⬆ STRAIGHT",
        "left": "◄ LEFT",
        "right": "RIGHT ►",
        "drift": " drift ",
        "nitro": " NITRO ",
        "back": " BACK "
    }
    text = action_texts.get(action, "")
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = cx - text_w // 2
        text_y = cy + radius + 50
        
        # Text background
        cv2.rectangle(image, (text_x - 10, text_y - text_h - 10), 
                     (text_x + text_w + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, primary_color, thickness)
    
    return image


def game():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')

    # Create the hand landmark detector using new Tasks Vision API
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cam_w, cam_h = 640, 480
    cap.set(3, cam_w)
    cap.set(4, cam_h)
    
    steering_angle = 0
    target_angle = 0

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.resize(image, (cam_w, cam_h))
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, _ = image.shape

        # Convert to RGB for mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Get frame timestamp in milliseconds
        timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        results = detector.detect_for_video(mp_image, timestamp_ms)

        co = []
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw all landmarks smoothly
                for i, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * imageWidth)
                    y = int(landmark.y * imageHeight)
                    size = 3 if i < 5 else 2
                    cv2.circle(image, (x, y), size, (0, 255, 100), -1)

                # Get wrist landmark (index 0)
                if len(hand_landmarks) > 0:
                    wrist = hand_landmarks[0]
                    co.append([int(wrist.x * imageWidth), int(wrist.y * imageHeight)])

        action = "straight"
        
        if len(co) == 2:
            xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
            
            # Calculate angle between hands for steering
            dx = co[1][0] - co[0][0]
            dy = co[1][1] - co[0][1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calculate rotation angle
            if dx != 0:
                base_angle = math.degrees(math.atan(abs(dy) / abs(dx)))
            else:
                base_angle = 90
            
            # Drift - Both hands at the top of the frame
            drift = False
            vertical_top_threshold = int(imageHeight / 3)
            if co[0][1] < vertical_top_threshold and co[1][1] < vertical_top_threshold:
                print("Drift")
                drift = True
                action = "drift"
                keyinput.release_key('w')
                keyinput.press_key('s')
                target_angle = -45 if co[0][0] < co[1][0] else 45

            # Nitro - Both hands at the bottom of the frame
            vertical_threshold = int(2 * imageHeight / 3)
            if co[0][1] > vertical_threshold and co[1][1] > vertical_threshold:
                print("Nitro.")
                action = "nitro"
                keyinput.press_key('space')
            else:
                keyinput.release_key('space')

            # Turn controls based on hand positions
            if co[0][0] > co[1][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 50:
                print("Turn left.")
                action = "left"
                if not drift:
                    keyinput.release_key('s')
                keyinput.release_key('d')
                keyinput.press_key('a')
                target_angle = -min(90, base_angle)

            elif co[1][0] > co[0][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 50:
                print("Turn left.")
                action = "left"
                if not drift:
                    keyinput.release_key('s')
                keyinput.release_key('d')
                keyinput.press_key('a')
                target_angle = -min(90, base_angle)

            elif co[0][0] > co[1][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 50:
                print("Turn right.")
                action = "right"
                if not drift:
                    keyinput.release_key('s')
                keyinput.release_key('a')
                keyinput.press_key('d')
                target_angle = min(90, base_angle)

            elif co[1][0] > co[0][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 50:
                print("Turn right.")
                action = "right"
                if not drift:
                    keyinput.release_key('s')
                keyinput.release_key('a')
                keyinput.press_key('d')
                target_angle = min(90, base_angle)

            else:
                if not drift:
                    print("keeping straight")
                    action = "straight"
                    keyinput.release_key('s')
                    keyinput.release_key('a')
                    keyinput.release_key('d')
                    keyinput.press_key('w')
                target_angle = 0

            # Smooth angle transition
            steering_angle = steering_angle * 0.85 + target_angle * 0.15
            
            # Draw the beautiful steering wheel
            wheel_radius = min(140, distance // 2 + 50)
            draw_steering_wheel(image, (xm, ym), wheel_radius, steering_angle, co, action)

        elif len(co) == 1:
            print("keeping back")
            action = "back"
            keyinput.release_key('a')
            keyinput.release_key('d')
            keyinput.release_key('w')
            keyinput.press_key('s')
            
            # Draw single hand indicator
            hx, hy = co[0]
            cv2.circle(image, (hx, hy), 30, (100, 100, 100), 3)
            cv2.putText(image, "BACK", (hx - 30, hy + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        else:
            keyinput.release_key('a')
            keyinput.release_key('d')
            keyinput.release_key('w')
            keyinput.release_key('s')
            
            # Draw "No hands detected" message
            cv2.putText(image, "No hands detected", (imageWidth // 2 - 100, imageHeight // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Draw status bar at bottom
        status_y = imageHeight - 30
        cv2.rectangle(image, (0, status_y), (imageWidth, imageHeight), (20, 20, 20), -1)
        
        # Display FPS
        fps_text = f"FPS: {int(cv2.getTickFrequency() / max(1, cv2.getTickCount() - timestamp_ms))}"
        cv2.putText(image, fps_text, (10, status_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Display action
        cv2.putText(image, f"Action: {action.upper()}", (imageWidth - 150, status_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        cv2.imshow("Hand Controller - Steering", image)
        cv2.setWindowProperty("Hand Controller - Steering", cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
