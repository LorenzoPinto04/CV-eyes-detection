## WEBCAM BLINK DETECTION

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

detector = dlib.get_frontal_face_detector()


# download pretrained model from: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
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


# treshold ratio that trigger the counter
threshold = .2

# number of sequential frames to trigger the counter
n_frame = 1


while keep == True:
    print("[INFO] start scanning. Press 'q' in your keyboard to stop.")
    start_time = time.time() # start time of the loop
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    rects = detector(frame, 1)
    copy = frame.copy()
    for i, rect in enumerate(rects):
        try:
            cv2.rectangle(copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255,255,0), 2)
            cv2.putText(copy, "Face #{}".format(i + 1), (rect.left(), rect.top()-10), cv2.FONT_HERSHEY_SIMPLEX, .3 ,(255, 255, 0), 1)
        except:
            continue
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
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
                blink += 1 
        else: 
            counter = 0 
        cv2.putText(copy, "Blinks:{}".format(str(blink)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .5 ,(0, 255, 0), 1)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 1200, 700)
    cv2.imshow('Frame', copy)
    #print("FPS: ", 1.0 / (time.time() - start_time))
    #print(blink)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] stop scanning.")
        break

vs.stop()
cv2.destroyAllWindows()
