# import the necessary packages
import numpy as np
import cv2
import time
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# # the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640,480))

while(True):
    count=0
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    i=0
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
        i+=1
        count=i      
        cv2.putText(frame, 'face num' + str(i), (xA - 10, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            
        
    # # Write the output video 
    # out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    print("no.of faces",i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)