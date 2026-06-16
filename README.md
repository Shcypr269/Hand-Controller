# Hand Gesture Game Controller

This project provides a computer vision-based control system that translates hand gestures into keyboard and mouse inputs. By utilizing your webcam, you can play games and control your computer without physical contact with your input devices. The project is powered by Python, OpenCV, and MediaPipe.

## Features

- **Car Game Controller**: Simulates a steering wheel using two hands. Allows you to steer, accelerate, reverse, drift, and activate nitro.
- **Pointer Controller**: Controls the mouse cursor position using one finger and allows clicking with two fingers.
- **Left Mouse Button Holder**: Allows holding the left mouse button down by raising a specific number of fingers.
- **On-Screen Visuals**: Displays a dynamic virtual steering wheel, hand tracking points, and current action states directly on the camera feed.

## Prerequisites

Before running the project, ensure you have Python installed. Install the required dependencies using the provided requirements file:

```bash
pip install -r controller/requirements.txt
```

Required packages:
- opencv-python
- mediapipe
- numpy
- mouse
- PyAutoGUI
- protobuf (version < 4 for compatibility)

## Usage

1. Navigate to the `controller` directory:
   ```bash
   cd controller
   ```

2. Run the main script to start the controller:
   ```bash
   python main.py
   ```

Note: By default, the `main.py` script starts the Car Game Controller. You can switch to other control modes (Pointer or Mouse Button Holder) by modifying the uncommented functions within `main.py`.

## Car Game Controls Reference

The Car Game Controller uses the relative positions of both hands to simulate a steering wheel and pedals.

| Hand Gesture                     | Action      | Keys Sent         |
|----------------------------------|-------------|-------------------|
| 2 hands, level                   | Go straight | W (accelerate)    |
| 2 hands, tilted left             | Turn left   | A (steer left)    |
| 2 hands, tilted right            | Turn right  | D (steer right)   |
| 2 hands, both at top of frame    | Drift       | S (brake/reverse) |
| 2 hands, both at bottom of frame | Nitro       | Space             |
| 1 hand detected                  | Go back     | S (reverse)       |
| No hands                         | Release all | All keys released |

To stop the program, make sure the camera window is active and press the "q" key.