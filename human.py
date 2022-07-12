import cv2 
import imutils 
   
# Initializing the HOG person 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
# Reading the Image 
image = cv2.imread('hog-detector.png') 
   
# Resizing the Image 
image1 = imutils.resize(image, width=min(500, image.shape[1])) #, image.shape[1]
   
# Detecting all humans 
(humans, _) = hog.detectMultiScale(image1, winStride=(5, 5), padding=(3, 3), scale=1.21)
# getting no. of human detected
print('Human Detected : ', len(humans))
   
# Drawing the rectangle regions
for (x, y, w, h) in humans: 
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2) 
  
# Displaying the output Image 
cv2.imshow("Image", image1) 
cv2.waitKey(1) 
   
cv2.destroyAllWindows() 