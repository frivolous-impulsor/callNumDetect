import cv2 as cv

picPath = 'Resources/librarians.jpg'
librarians = cv.imread(picPath)
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + cascPath)

def resizeRatio(frame, scale=0.75):
    width = int(librarians.shape[1] * scale)
    height = int(librarians.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

librarians = resizeRatio(librarians, 0.2)

grayLibs = cv.cvtColor(librarians, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayLibs, scaleFactor = 1.1,
                                      minNeighbors = 6)


print(len(faces))
for (x, y, w, h) in faces:
    cv.rectangle(librarians, (x,y), (x+w, y+h), (0, 255, 0), 2)
cv.imshow('detected libs', librarians)
cv.waitKey(0)