import cv2

# pre_trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture webcam.

webcam = cv2.VideoCapture(0)
# 0 FOR video capture can use video instead of 0 ('video.mp4')

while True:
    successful_frame_read, frame = webcam.read()

    # convert video/ webcam to grayscale
    grayscale_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cord = trained_face_data.detectMultiScale(grayscale_vid)

    # draw Rectangles around the faces

    for (x, y, w, h) in face_cord:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # (img, (TopLeft cord), (cord+width), (BGR col), thickness)

    cv2.imshow('webcam face detect', frame)
    key = cv2.waitKey(1)

    # quit at q or Q button
    if key == 81 or key == 113:
        break
