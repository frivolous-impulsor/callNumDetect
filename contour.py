import cv2 as cv

dogs = cv.imread('Resources/dogs.jpg')
cv.imshow('dogs', dogs)

grayDogs = cv.cvtColor(dogs, cv.COLOR_BGR2GRAY)
cv.imshow('gray dogs', grayDogs)

blurDogs = cv.GaussianBlur(grayDogs, (9,9), 0)
cv.imshow('blur dogs', blurDogs)

cannyDogs = cv.Canny(blurDogs, 80, 100)
cv.imshow('canny dogs', cannyDogs)

contours, hierarchy = cv.findContours(cannyDogs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv.drawContours(dogs, contours, -1, (0, 255, 0), 2)
cv.imshow("contoured dogs", dogs)

cv.waitKey(0)