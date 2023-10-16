import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm

p_time = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    if len(lm_list):
        print(lm_list[4])
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,  (255, 255, 0), 3)
    cv2.imshow('img', img)
    cv2.waitKey(1)