# import open cv module
import cv2

# dataset load
trainedData=cv2.CascadeClassifier('haarcascades.xml')

# start the webcam
webcam=cv2.VideoCapture(0)

while True:
    success,frame=webcam.read()

    # conversion to gray scale
    grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect faces
    facecoordinates=trainedData.detectMultiScale(grayimg)
    # print(facecoordinates)

    #  [209 124  48  48]]    coordinates of rectangle to detect face
    # [x,y,w(width),h(height)]

    for x,y,w,h in facecoordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #  display video
    cv2.imshow('single image',frame)

    # pause execution of program until any key is pressed
    key=cv2.waitKey(1)    # each image will shift to another image im 1 millisecond

    if key==27: # ASCII value of escape is 27
        break

webcam.release() # close webcam