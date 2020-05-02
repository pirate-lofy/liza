from deepface.commons.realtime import Face
import cv2 as cv

face=Face()
cap=cv.VideoCapture(0)
c=0
while True:
    _,frame=cap.read()
    frame,p=face.process(frame,c=c)
    if p is not None:
        print(p)
    cv.imshow('',frame)
    if cv.waitKey(1)&0xFF==27:
        break
    c+=1

cap.release()
cv.destroyAllWindows()