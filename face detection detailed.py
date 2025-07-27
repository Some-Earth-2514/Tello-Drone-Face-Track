# Import required modules for object tracking, computer vision, drone control, and system operations
from pyimagesearch.objcenter import ObjCenter  # Custom class to detect object (e.g., face) center
import cv2  # OpenCV for image processing
from pyimagesearch.pid import PID  # Custom PID controller class
from djitellopy import Tello  # DJI Tello drone SDK
import signal  # For capturing interrupt signals (Ctrl+C)
import sys  # For exiting the program cleanly
import imutils  # For easier OpenCV image processing

"""
This script will just turn on the video feed and WILL NOT set the drone in flight.
It is strictly meant to test the video feed and the ability to detect faces.
"""

# Instantiate a Tello drone object
tello = Tello()

# Define a function to safely shut down the stream and land the drone on keyboard interrupt (Ctrl+C)
def signal_handler(sig, frame):
    tello.streamoff()  # Turn off the video stream
    tello.land()       # Ensure the drone lands safely (in case it was flying)
    sys.exit()         # Exit the program

# Bind the signal handler to SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Connect to the Tello drone
tello.connect()

# Turn on the video stream
tello.streamon()

# Access the video stream reader object
frame_read = tello.get_frame_read()

# Initialize the object (face) center detection using Haar cascade classifier
face_center = ObjCenter("./haarcascade_frontalface_default.xml")

# Create PID controllers for pan (left-right) and tilt (up-down) tracking
pan_pid = PID(kP=0.7, kI=0.0001, kD=0.09)
tilt_pid = PID(kP=0.7, kI=0.0001, kD=0.09)

# Initialize the PID controllers (resets integral and derivative terms)
pan_pid.initialize()
tilt_pid.initialize()

run_pid = True  # Flag to control whether to run PID tracking logic

# Infinite loop to process video frames
while True:
    # Grab the latest frame from the drone's camera
    frame = frame_read.frame

    # Resize the frame to a fixed width of 400 pixels for faster processing
    frame = imutils.resize(frame, width=400)
    H, W, _ = frame.shape  # Get height and width of the resized frame

    # Calculate center of the frame (target location to align the face to)
    centerX = W // 2
    centerY = H // 2

    # Draw a red circle at the center of the frame
    cv2.circle(frame, center=(centerX, centerY), radius=5, color=(0, 0, 255), thickness=-1)

    # Detect the location of a face in the frame
    frame_center = (centerX, centerY)
    objectLoc = face_center.update(frame, frameCenter=None)

    # Extract the coordinates of the detected face center, the bounding box, and movement delta
    ((objX, objY), rect, d) = objectLoc

    # Skip this frame if the detected object's position is too far from the previous (reduces jitter)
    if d > 50:
        continue

    # If a face was detected (i.e., bounding box is not None)
    if rect is not None:
        (x, y, w, h) = rect  # Get face bounding box coordinates

        # Draw a green rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a blue circle at the center of the detected face
        cv2.circle(frame, center=(objX, objY), radius=5, color=(255, 0, 0), thickness=-1)

        # Draw an arrowed line from frame center to face center to visualize error direction
        cv2.arrowedLine(frame, frame_center, (objX, objY), color=(0, 255, 0), thickness=2)

        # If PID tracking is enabled
        if run_pid:
            # Calculate pan error (how far face is from horizontal center)
            pan_error = centerX - objX
            pan_update = pan_pid.update(pan_error, sleep=0)  # Get PID correction

            # Calculate tilt error (how far face is from vertical center)
            tilt_error = centerY - objY
            tilt_update = tilt_pid.update(tilt_error, sleep=0)  # Get PID correction

            # Print out the error and PID outputs for debugging
            print(pan_error, int(pan_update), tilt_error, int(tilt_update))

            # Overlay pan PID values on the frame
            cv2.putText(frame, f"X Error: {pan_error} PID: {pan_update:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Overlay tilt PID values on the frame
            cv2.putText(frame, f"Y Error: {tilt_error} PID: {tilt_update:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the processed video frame in a window
    cv2.imshow("Face Tracking", frame)

    # Wait briefly for a key press (1ms); allows for smooth video display
    cv2.waitKey(1)
