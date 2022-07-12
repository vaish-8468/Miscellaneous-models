import time

import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
 
while True:
    count = 0
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    i = 0
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        i += 1
        count = i
        cv2.putText(frame, 'face num' + str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)
    cv2.imshow('frame', frame)
    print('total_number_of_faces_detected', count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(5)


cap.release()
cv2.destroyAllWindows()
