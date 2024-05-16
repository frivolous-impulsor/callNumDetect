import cv2 as cv
import sys

cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + cascPath)

if len(sys.argv) < 2:
    video = cv.VideoCapture(1)
else:
    video = cv.VideoCapture(sys.argv[1])

while True:
    ret, img = video.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    print(f"Number of Faces Found: {len(faces)}")

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv.imshow('detecting faces', img)

    if cv.waitKey(1) & 0xff == ord('d'):
        break

video.release()
cv.destroyAllWindows()
