"""
This script detects ArUco markers and estimates their poses using camera calibration values.
It displays the original image and the image with detected markers and their poses.
It requires OpenCV 4.7 or newer, including OpenCV 5, and uses ArucoDetector and cv2.solvePnP
(the older aruco.detectMarkers and aruco.estimatePoseSingleMarkers functions do not exist in OpenCV 5).

@author: ioannisgeorgilas
"""

import cv2
import cv2.aruco as aruco
import logging
import numpy as np
import socket
import time # We will use this to ensure a steady processing rate
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Load the camera calibration values
calibration_path = Path(__file__).resolve().parent / "workdir" / "Calibration.npz"
camera_calibration = np.load(calibration_path)
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera
logging.info("Loaded camera calibration from %s", calibration_path)

# Define the ArUco dictionary and parameters
marker_size = 40 # the real size of the printed marker in mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# The 3D position (in mm) of the 4 corners of a marker, in the marker's own coordinate system (origin in the centre).
# The order is the same as the corners returned by the detector: top-left, top-right, bottom-right, bottom-left
half = marker_size / 2
marker_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

# Define a processing rate in milliseconds (e.g., 0.25 means 4 frames per second).
processing_period = 0.25

# Create two OpenCV named windows
cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gray", cv2.WINDOW_AUTOSIZE)

# Position the windows next to each other
cv2.moveWindow("Gray", 640, 100)
cv2.moveWindow("Frame", 0, 100)
# Start capturing video
camera_index = 1
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    logging.error("Could not open camera %s", camera_index)
    raise SystemExit(1)
logging.info("Camera %s opened", camera_index)

# Send [enable, command] as a two-byte UDP packet to Simulink.
simulink_address = ("138.38.226.145", 25000)
marker_states = {1: (0, 1), 5: (0, 0)}
stationary_state = (1, 0)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Set the starting time
start_time = time.time()
fps = 0
last_logged_marker_ids = None
last_sent_state = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera %s; exiting", camera_index)
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    marker_ids = tuple(sorted(int(marker_id) for marker_id in ids.flatten())) if ids is not None else ()
    enable, command = marker_states.get(marker_ids[0], stationary_state) if len(marker_ids) == 1 else stationary_state
    state = (enable, command)
    udp_socket.sendto(bytes(state), simulink_address)
    if state != last_sent_state:
        logging.info("Sent Simulink values: enable=%s, command=%s", enable, command)
        last_sent_state = state

    if marker_ids != last_logged_marker_ids:
        if marker_ids:
            logging.info("Detected ArUco marker IDs: %s", marker_ids)
        elif last_logged_marker_ids:
            logging.info("No ArUco markers detected")
        last_logged_marker_ids = marker_ids

    # If markers are detected
    # (when nothing is detected ids is None in OpenCV 4 but an empty array in OpenCV 5, so we check the corners)
    if len(corners) > 0:
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate the pose of each marker from its 4 detected corners
        for i in range(len(corners)):
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                corners[i].reshape(4, 2),
                CM,
                dist_coef,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                marker_id = int(ids[i].item())
                tx, ty, tz = tvec.flatten()
                label = f"ID {marker_id}: t=({tx:.1f}, {ty:.1f}, {tz:.1f}) mm"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(label, font, 0.5, 1)
                marker_corners = corners[i].reshape(4, 2)
                x = int(np.min(marker_corners[:, 0]))
                y = int(np.min(marker_corners[:, 1])) - 8
                x = max(0, min(x, frame.shape[1] - text_size[0] - 4))
                y = max(text_size[1] + 4, y)
                cv2.putText(frame, label, (x, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (x, y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Draw axis for each marker (axis length 10 mm)
                frame = cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 10)


    # Add the frame rate to the image
    cv2.putText(frame, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ensure a steady processing rate
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()



# When everything is done, release the capture and close windows
udp_socket.sendto(bytes(stationary_state), simulink_address)
udp_socket.close()
cap.release()
cv2.destroyAllWindows()
logging.info("Camera capture stopped")