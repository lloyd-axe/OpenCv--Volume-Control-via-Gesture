import cv2 as cv
import numpy as np
import mediapipe as mp

class Tracker:
    def __init__(
        self,
        detectionCon = 0.5,
        trackingCon = 0.5,):
        self.holistic = mp.solutions.holistic
        self.hands = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils
        self.holistic_model = self.holistic.Holistic(min_detection_confidence = detectionCon,min_tracking_confidence = trackingCon)
    
    def Detect(
        self,
        frame,
        color = cv.COLOR_BGR2RGB):
        return self.holistic_model.process(cv.cvtColor(frame, color))

    def DetectHands(
        self, 
        frame, 
        color=cv.COLOR_BGR2RGB):
        hand = self.hands.Hands()
        return hand.process(cv.cvtColor(frame, color))

    def DrawLandMarks(
        self,
        frame,
        results,
        head = True,
        pose = True,
        lh = True,
        rh = True):
        if head:
            self.drawing.draw_landmarks(frame,
                                      results.face_landmarks,
                                      self.holistic.FACEMESH_TESSELATION,
                                      self.drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      self.drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
        if pose:
            self.drawing.draw_landmarks(frame, 
                                      results.pose_landmarks, 
                                      self.holistic.POSE_CONNECTIONS)
        if lh:
            self.drawing.draw_landmarks(frame, 
                                      results.left_hand_landmarks, 
                                      self.holistic.HAND_CONNECTIONS)
        if rh:
            self.drawing.draw_landmarks(frame, 
                                      results.right_hand_landmarks, 
                                      self.holistic.HAND_CONNECTIONS)

    def ExtractKeyPoints(
        self, 
        results):
        keyPoints = {'pose': [], 'head': [], 'left': [], 'right': []}
        keyPoints['pose'] = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132*4)
        keyPoints['head'] = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        keyPoints['left'] = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        keyPoints['right'] = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate(keyPoints)
    
    def FindPositions(
        self,
        results):
        landmarks = {'pose': [], 'head': [], 'left': [], 'right': []}
        landmarks['pose'] = [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark] if results.pose_landmarks else None
        landmarks['head'] = [[res.x, res.y, res.z] for res in results.face_landmarks.landmark] if results.face_landmarks else None
        landmarks['left'] = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else None
        landmarks['right'] = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else None
        return landmarks

    def CheckPosition(
        self, 
        frame, 
        positions, 
        size = 10, 
        color = (0,255,0)):
        for pos in positions:
            cv.circle(frame, (pos[0],pos[1]), size, color, cv.FILLED)
