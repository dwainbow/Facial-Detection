import cv2
import numpy as np
# image=cv2.imread("download.png") #shows image
# cv2.imshow("Output",image) #shows image
# cv2.waitKey(0) #how long image shows up for

# #Video Capture
# capture=cv2.VideoCapture(0) #0 sets defualt camera
# capture.set(3,640) #width
# capture.set(4,480) #height
# capture.set(10,100)#brightness
# while True:
#     success,image=capture.read() #success is a boolean
#     cv2.imshow("Video",image)
#     if cv2.waitKey(1) and 0xFF ==ord("q"): #adds delay and looks for q when pressed to break the loop
#         break
#
# #GreyScale
#
# image=cv2.imread("download.png")
# imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Set image to gray color
# imageBlur=cv2.GaussianBlur(imageGray,(7,7),0) #Blurs image
# cv2.imshow("image", imageBlur)
# cv2.waitKey(0)

# #How to draw on images
# image=np.zeros((512,512,3),np.int8) #np.int8 gives values from 0 to 255
# print(image)
# #image[:]=255,0,0 #[:] means entire image
# cv2.line(image,(0,0),(300,300),(0,255)) # makes line
# cv2.rectangle(image,(0,0),(300,300),(0,255))
# cv2.imshow("Window",image)
# cv2.waitKey(0)
# #.shape function checks the dimensions

#detect faces
faceDetector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image=cv2.imread("faces.jpeg") #shows image
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceDetector.detectMultiScale(imageGray,1.1,4) #numbers are scale factors
for x,y,w,h in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),2) #creates a box around every detected face


cv2.imshow("Output",image) #shows image
cv2.waitKey(0) #how long image shows up for

