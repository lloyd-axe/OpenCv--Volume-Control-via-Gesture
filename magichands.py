import time
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
'''
------------------------------------------------------------
This module will simply enable you to control your PC's
master volume without clicking anything. 

This version will only recognize the user's right hand.
Planning to add flexibility in the options in the future versions.
------------------------------------------------------------
'''
class magichands:
    def __init__(self,
        detect_conf = 0.5, 
        tracking_conf = 0.5):
        #Initiate camera and master volume instances
        self.capture = cv.VideoCapture(0) #PC camera
        self.devices = AudioUtilities.GetSpeakers()
        self.volume = cast(
            self.devices.Activate(
                IAudioEndpointVolume._iid_, 
                CLSCTX_ALL, 
                None), 
            POINTER(IAudioEndpointVolume))
        
        #Mediapipe variables
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence = detect_conf, 
            min_tracking_confidence = tracking_conf)
    
    def activate(self, 
        trigger = 40, 
        maxDist = 200, 
        minDist = 0, 
        minVolRange = -65, 
        maxVolRange = 0):
        print('magichands now activated!')
        print('Press Q to quit.')
        while True:
            checkColor = (0,0,255) #Change finger tip colors to RED
            _, frame = self.capture.read()
            h, w, c = frame.shape
            results = self.holistic_model.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            #Get right hand keypoints
            right_landmarks = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21,3))


            if np.array(right_landmarks).any():
                #Calculate distances
                tips = [right_landmarks[12], right_landmarks[8], right_landmarks[4]] #Find thumb, pointing and middle finger tips
                x0, y0 = int(tips[0][0]*w), int(tips[0][1]*h)
                x1, y1 = int(tips[1][0]*w), int(tips[1][1]*h)
                x2, y2 = int(tips[2][0]*w), int(tips[2][1]*h)
                
                '''
                Trigger Condition:
                1. Pointing and middle finger tips are close together
                2. Pointing and middle finger tips are aligned horizontally

                Finger tips will turn GREEN if trigger conditions are met. 
                If not, it will turn RED.

                Plan to improve trigger conditions in the future versions
                '''
                trigDist = math.hypot(x1-x0, y1-y0)
                if trigDist < trigger:
                    x01, y01 = (x1, y0)
                    angle = math.degrees(math.acos((math.hypot(x01-x0, y01-y0))/trigDist))
                    if -15 < angle < 90: # Determine if tips are horizontally aligned
                        dist = math.hypot(x2-x1, y2-y1) #Determine if tips are close
                        checkColor = (0,255,0) #Change finger tip colors to GREEN
                        #Calculate volume
                        volPerc = (dist * (minVolRange/(minDist - maxDist))) + ((-minVolRange*maxDist)/(minDist - maxDist))
                        if minVolRange <= volPerc <= maxVolRange:
                            self.volume.SetMasterVolumeLevel(volPerc, None)      
                #Visualize finger tips
                for pos in [(x0,y0), (x1,y1), (x2,y2)]:
                    cv.circle(frame, (pos[0],pos[1]), 5, checkColor, cv.FILLED)
            cv.imshow('Capture', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv.destroyAllWindows()


# -------------------------TEST-------------------------
if __name__ == "__main__":
    mh = magichands()
    mh.activate()