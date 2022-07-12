import cv2
import dlib
import time

import imutils

imagePath = cv2.VideoCapture(0)

while True:
    ret,image =imagePath.read()


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30))

    print("[INFO] Found {0} Faces!".format(len(faces)))

    for face in faces:
        import pdb
        pdb.set_trace()
        x, y = face.left(), face.top()
        w, h = face.right(), face.bottom()
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
        print(x, y, w, h)
    cv2.imshow('frame', image)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(5)

imagePath.release()
cv2.destroyAllWindows()