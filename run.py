from gazetracker.gaze import Gaze
from deepFace.deepface.commons.realtime import Face
from posestimator.head import Head
from facialdriving.drive import Driver
import cv2 as cv 


#gaze=Gaze()
#face=Face(emotions=True)
driver=Driver()
cap=cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 840)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv.CAP_PROP_FPS, 60)
_,frame=cap.read()
head=Head(frame)
        
    
while True:
    _,frame=cap.read()
    frame,rotation=head.process(frame)
    # frame=gaze.process(frame)
    #res=face.process(frame,emotions=True)
    # if res is not None:
    #     frame=res['e_frame']
    driver.process(rotation)
    cv.imshow('',frame)
    if cv.waitKey(1)&0xFF==27:
        break
    
cap.release()
cv.destroyAllWindows()

