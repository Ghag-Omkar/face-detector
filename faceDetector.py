import cv2
import random

# pre_trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img1 = cv2.imread('2.jpg')
img2 = cv2.imread('grp_test.jpg')


# convert img to grayscale
grayscale_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# detect face
face_cord = trained_face_data.detectMultiScale(grayscale_img)

# draw Rectangles around the faces


for (x, y, w, h) in face_cord:
    cv2.rectangle(img2, (x, y), (x + w, y + h), (random.randrange(128, 256), random.randrange(128, 256), random.randrange(128, 256)), 2)
# (img, (TopLeft cord), (cord+width), (BGR col), thickness)

cv2.imshow('Face Detector', img2)
cv2.waitKey()
