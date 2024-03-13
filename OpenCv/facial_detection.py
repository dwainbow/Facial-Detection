import cv2
import numpy as np

#detect faces
faceDetector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image=cv2.imread("faces.jpeg") #shows image
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceDetector.detectMultiScale(imageGray,1.1,4) #numbers are scale factors
for x,y,w,h in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),2) #creates a box around every detected face


cv2.imshow("Output",image) #shows image
cv2.waitKey(0) #how long image shows up for

