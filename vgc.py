import time
import math
import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from MediaPipeModule import Tracker
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VGC:
    def __init__(self):
        self.capture = cv.VideoCapture(0)
        self.tracker = Tracker()
        self.devices = AudioUtilities.GetSpeakers()
        self.volume = cast(
            self.devices.Activate(
                IAudioEndpointVolume._iid_, 
                CLSCTX_ALL, 
                None), 
            POINTER(IAudioEndpointVolume))
    
    def Activate(
        self, 
        trigger = 40, 
        maxDist = 200, 
        minDist = 0, 
        minVolRange = -65, 
        maxVolRange = 0):
        while True:
            checkColor = (0,255,0)
            _, frame = self.capture.read()
            h, w, c = frame.shape
            results = self.tracker.Detect(frame)

            #get keypoints for volume control
            landmarks = self.tracker.GetPositions(results)['right'] #only right hand
            if np.array(landmarks).any():
                tips = [landmarks[12], landmarks[8], landmarks[4]]
                x0, y0 = int(tips[0][0]*w), int(tips[0][1]*h)
                x1, y1 = int(tips[1][0]*w), int(tips[1][1]*h)
                x2, y2 = int(tips[2][0]*w), int(tips[2][1]*h)
                
                #Trigger condition
                #Control volumn only if pointing and middle finger are together and is horizontal
                trigDist = math.hypot(x1-x0, y1-y0)
                if trigDist < trigger:
                    x01, y01 = (x1, y0)
                    angle = math.degrees(math.acos((math.hypot(x01-x0, y01-y0))/trigDist))
                    if -15 < angle< 90:
                        dist = math.hypot(x2-x1, y2-y1)
                        checkColor = (0,0,255)

                        #Calculate volume
                        volPerc = (dist * (minVolRange/(minDist - maxDist))) + ((-minVolRange*maxDist)/(minDist - maxDist))
                        if minVolRange <= volPerc <= maxVolRange:
                            self.volume.SetMasterVolumeLevel(volPerc, None)
                #for visualization
                self.tracker.CheckPosition(
                    frame,
                    [(x0,y0), (x1,y1), (x2,y2)], 
                    5, 
                    checkColor) 
            cv.imshow('Capture', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv.destroyAllWindows()

test = VGC()
test.Activate()