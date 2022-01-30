import time
import math
import cv2 as cv
import numpy as np
from matplotlib.cbook import maxdict
from MediaPipeModule import Tracker
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

tracker = Tracker()

###########################
maxDist = 230
minDist = 10
pTime = 0
###########################

cap = cv.VideoCapture(0)

#pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
barVol = 0
while cap.isOpened():
    isTrue, frame = cap.read()
    h, w, c = frame.shape
    if not isTrue:
        break

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Detect
    results = tracker.Detect(frame)
    lm = tracker.FindPositions(results)['right']
    if lm:
        pointingThumbs = [lm[8], lm[4]]
        x0, y0 = int(pointingThumbs[0][0]*w), int(pointingThumbs[0][1]*h)
        x1, y1 = int(pointingThumbs[1][0]*w), int(pointingThumbs[1][1]*h)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        tracker.CheckPosition(frame,[(x0,y0), (x1,y1)], 5, (0,255,0))
        cv.line(frame, (x0,y0), (x1,y1), (0,0,255), 3)
        dist = math.hypot(x1-x0, y1-y0)
        volPerc = (dist * (-65/(minDist - maxDist))) + ((65*maxDist)/(minDist - maxDist))
        barVol = np.interp(dist, [50,300], [400, 150])
        if -65 <= volPerc <= 0:
            print(volPerc)
            volume.SetMasterVolumeLevel(volPerc, None)
    #Visual
    #cv.rectangle(frame, (50,150), (85, 400), (0,255,0), 3)
    #cv.rectangle(frame, (50,int(barVol)), (85, 400), (0,255,0), cv.FILLED)

    cv.putText(
        frame, 
        f'FPS: {int(fps)}',
        (20,40), 
        cv.FONT_HERSHEY_COMPLEX,
        0.5, 
        (0,0,255),
        1)
    #cv.imshow('Capture', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
