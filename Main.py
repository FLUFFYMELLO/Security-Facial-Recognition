import cv2
import face_recognition
import json
import streamlit as st



Camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

while True:
    ret, frame = Camera.read()
    Grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Faces = detector.detectMultiScale(Grayscale, 1.5, 5)
    for (x,y,w,h) in Faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == ord('p'):
        break

Camera.release()
cv2.destroyAllWindows()