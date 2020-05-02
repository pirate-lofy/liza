import cv2 as cv
import numpy as np
import requests as rq

url='http://192.168.1.2:8080/shot.jpg'

while True:
    img=rq.get(url)
    img=np.array(bytearray(img.content),dtype=np.uint8)
    img=cv.imdecode(img,-1)
    
    cv.imshow('',img)
    if cv.waitKey(1)==27:
        break
    
    