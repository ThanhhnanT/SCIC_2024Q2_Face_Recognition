from scrfd import SCRFD, Threshold
from PIL import Image
import cv2
from pprint import pprint

face_detector = SCRFD.from_path("Scrfd/scrfd.onnx")
threshold = Threshold(probability=0.4)

cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if not flag:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    faces = face_detector.detect(image, threshold=threshold)
    for face in faces:
        bbox = face.bbox
        x_left = int( bbox.upper_left.x)
        y_left = int(bbox.upper_left.y)
        x_right = int(bbox.lower_right.x)
        y_right = int(bbox.lower_right.y)
        frame = cv2.rectangle(frame, (x_left, y_left), (x_right, y_right), (255,0,0), 1)
        frame = frame[y_left:y_right, x_left:x_right]


    cv2.imshow('webcam', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
