
import numpy as np
import cv2
import math
import pyautogui
import time

# Open Camera
capture = cv2.VideoCapture(0)


mute_counter = 0
stop_counter = 0
play_counter = 0;

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    cv2.imshow("Thresholded", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        hull = cv2.convexHull(contour)

        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

           
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Print number of fingers
        if count_defects == 0:
            play_counter = play_counter+1
            if play_counter > 50:
                cv2.putText(frame, "Play", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
                #pyautogui.hotkey('volumeup')
                pyautogui.hotkey('playpause')
                play_counter = 0
                stop_counter = 0
                mute_counter = 0
                #time.sleep(0.5)
        elif count_defects == 1:
            stop_counter = stop_counter +1;
            if stop_counter > 100:
                stop_counter = 0
                mute_counter = 0
                play_counter = 0
                cv2.putText(frame, "Stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
                #pyautogui.hotkey('volumedown')
                pyautogui.hotkey('stop')
        elif count_defects == 2:
            mute_counter = mute_counter+1
            if mute_counter > 125:
                mute_counter = 0
                stop_counter = 0
                play_counter = 0
                cv2.putText(frame, "Mute", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
                pyautogui.hotkey('volumemute')
            #time.sleep(0.1)
        elif count_defects == 3:
            cv2.putText(frame, "Volume up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            #pyautogui.hotkey('playpause')
            pyautogui.hotkey('volumeup')
        elif count_defects == 4:
            cv2.putText(frame, "volume down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            #pyautogui.hotkey('stop')
            pyautogui.hotkey('volumedown')
        else:
            pass
    except:
        pass

    cv2.imshow("Gesture", frame)
    # all_image = np.hstack((drawing, crop_image))
    # cv2.imshow('Contours', all_image)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()