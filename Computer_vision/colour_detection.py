import cv2
import numpy as np

# Open webcamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error, could not open webcamera.")
    exit()

print("Program is running. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read image from camera.")
        break

    # RGB/BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking (Focus on RED colour: from 0-10 to 170-180)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combining two masks to get the final mask for red color
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Noise filtering
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the red objects
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            # Moments for center (u, v)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                
                # Drawing circle around the detected object
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

                # Center of the object
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({u}, {v})", (u + 10, v - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show 2 windows: Original and Masked
    cv2.imshow("Webcam - Original", frame)
    cv2.imshow("Red Mask", red_mask)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()