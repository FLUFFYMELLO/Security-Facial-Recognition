import cv2

Camera = cv2.VideoCapture(0)

while True:
    ret, frame = Camera.read()