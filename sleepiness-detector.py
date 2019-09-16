## WEBCAM SLEEPINESS DETECTION

from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import sounddevice as sd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


vs = VideoStream(src=0).start()
time.sleep(2.0)
blink = 0
counter = 0 
keep = True
fps_start = 10
fps = fps_start


# sound for alarm
# Samples per second
sps = 44100
freq_hz = 1000.0
# calculate the waveform
each_sample_number = np.arange(sps)
waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)



# treshold ratio that trigger the counter
threshold = .22

# seconds to trigger alarm
time_alarm = 2

print("[INFO] start scanning. Press 'q' in your keyboard to stop.")
while keep == True:
    n_frame = round(time_alarm * fps)
    start_time = time.time() # start time of the loop
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    rects = detector(frame, 1)
    copy = frame.copy()
    for i, rect in enumerate(rects):
        cv2.rectangle(copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255,255,0), 2)
        cv2.putText(copy, "Face #{}".format(i + 1), (rect.left(), rect.top()-10), cv2.FONT_HERSHEY_SIMPLEX, .3 ,(255, 255, 0), 1)
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        eye_sx = cv2.convexHull(shape[36:41])
        cv2.drawContours(copy, [eye_sx], 0, (0, 255, 255), 1)
        eye_dx = cv2.convexHull(shape[42:47])
        cv2.drawContours(copy, [eye_dx], 0, (0, 255, 255), 1)
        #mouth = cv2.convexHull(shape[48:67])
        #cv2.drawContours(copy, [mouth], 0, (255, 255, 0), 2)
        ratio_sx = (distance.euclidean(
            totuple(shape[37]), totuple(shape[41])
        ) + distance.euclidean(
            totuple(shape[38]), totuple(shape[40]))
                   ) / (2 * distance.euclidean(
            totuple(shape[36]), totuple(shape[39])))
        ratio_dx = (distance.euclidean(
            totuple(shape[43]), totuple(shape[47])
        ) + distance.euclidean(
            totuple(shape[44]), totuple(shape[46]))
                   ) / (2 * distance.euclidean(
            totuple(shape[42]), totuple(shape[45])))        
        ratio = (ratio_sx + ratio_dx) / 2
        if ratio <= threshold:
            counter += 1
            if counter == n_frame:
                sd.play(waveform, sps)
        else:
            counter = 0
            sd.stop()
        cv2.putText(copy, "Treshold: {}%".format(counter * 100 / n_frame), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, .5 ,(255, 255, 0), 1)
        cv2.putText(copy, "Sleepiness: {}".format(str(ratio)[:4]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5 ,(255, 255, 0), 1)
    fps = round(1.0 / (time.time() - start_time))
    cv2.putText(copy, "FPS: {}".format(fps), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .5 ,(255, 255, 0), 1)
    cv2.imshow('Frame', copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] stop scanning.")
        break

vs.stop()
cv2.destroyAllWindows()
