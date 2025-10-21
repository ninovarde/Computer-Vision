# Motion Detection Alarm System 

"""
This script implements a basic Motion Detection Alarm System.
"""


# Import libraries
import threading # Allows to run multiple operations in parallel
import winsound  # Windows-only module for simple sound alerts


import cv2      # The main library used for computer vision tasks
import imutils  # Convenience library that simplifies some OpenCV operations


# Intialize the webcam and set the desired resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


_, start_frame = cap.read()

start_frame = imutils.resize(start_frame, width = 500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, ksize = (5,5), sigmaX=0)


"""

The Gaussian Blur smooths the image and helps reduce noise and small 
variations, so that only significant motion is detected.

- `ksize` sets the intensity of the blurring process.
- `sigmaX` is the standard deviation in the X direction. If `sigmaX = 0`, 
   OpenCV automatically calculates it.
- `sigmaY`, if not specified, is set equal to `sigmaX`.

"""

alarm = False
alarm_mode = False
alarm_counter = 0

alarm_sound_frequency = 2500  # Hz
alarm_sound_duration = 1000   # ms 

motion_intensity_sensitivity = 100
motion_duration_sensitivity = 5
                    




def beep_alarm(
        frequency = alarm_sound_frequency, 
        duration = alarm_sound_duration
        ):
    
    global alarm 
    
    print("ALARM")
    winsound.Beep(frequency, duration)
    
    alarm = False
    


print("\n\nCOMMANDS: \n\n't' activate/deactivate motion detection\n'q' Quit")
while True: 
    _, frame = cap.read()
    frame = imutils.resize(frame, width = 500)
    
    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
        
        difference = cv2.absdiff(frame_bw, start_frame) 
        threshold = cv2.threshold(
                                src = difference, 
                                thresh = 25, 
                                maxval = 255, 
                                type = cv2.THRESH_BINARY
                                )[1]
        
        start_frame = frame_bw
        
        if threshold.sum() > motion_intensity_sensitivity: 
            alarm_counter +=1
        else:
            if alarm_counter >0:
                alarm_counter -=1
        
        cv2.imshow("Cam", threshold)
    
    else:
        cv2.imshow("Cam",frame)
        
    
    if alarm_counter>motion_duration_sensitivity:
        alarm_counter = motion_duration_sensitivity
        if not alarm:
            alarm = True
            threading.Thread(target = beep_alarm).start()
            
    key_pressed = cv2.waitKey(1)
    
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode 
        alarm_counter = 0
            
    if key_pressed == ord("q"):
        alarm_mode = False
        break

cap.release()
cv2.destroyAllWindows()
    












