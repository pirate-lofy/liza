import cv2 as cv
import numpy as np
import pandas as pd

from deepface.extendedmodels import Emotion,Age, Gender
from deepface.commons import functions

class Face:
    #------global variables
    pivot_img_size = 112
    input_shape = (224, 224)
    text_color = (255,255,255)
    pivot_img_size = 112
    a=e=g=r=False
    
    def __init__(self,emotions=False,age=False,gender=False,facial_recognition=False):
        opencv_path = functions.get_opencv_path()
        face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
        self.face_cascade = cv.CascadeClassifier(face_detector_path)
       	
        # load models
        if emotions:
            self.e=True
            self.emotion_model = Emotion.loadModel()
        if age:
            self.a=True
            self.age_model = Age.loadModel()
        if gender:
            self.g=True
            self.gender_model = Gender.loadModel()
        
        print('Face Class initiallized successfully')
        
    
    def draw_transparenc(self,frame,dims,resx):
        x,y,w,h=dims
        overlay = frame.copy()
        opacity = 0.4
        if x+w+self.pivot_img_size < resx:
        	#right
            cv.rectangle(frame
                         , (x+w,y)
                          , (x+w+self.pivot_img_size, y+h)
                          , (64,64,64),cv.FILLED)
            cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        elif x-self.pivot_img_size > 0:
            #left
            cv.rectangle(frame
                   , (x-self.pivot_img_size,y)
                   , (x, y+h)
                   , (64,64,64),cv.FILLED)
            cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame
    
    
    def draw_emotions(self,frame,emotion_df,dims,resx):
        x,y,w,h=dims
        for index,instance in emotion_df.iterrows():
            emotion_label='%s'%(instance.emotion)
            emotion_score=instance.score/100.
            bar_x=int(35*emotion_score)
            
            if x+w+self.pivot_img_size<resx:
                text_location_y=y+20+(index+1)*20
                text_location_x=x+w
                
                if text_location_y<y+h:
                    cv.putText(frame,emotion_label,
                               (text_location_x,text_location_y), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv.rectangle(frame
                                  , (x+w+70, y + 13 + (index+1) * 20)
                                  , (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                  , (255,255,255), cv.FILLED)
            
            elif x-self.pivot_img_size > 0:
                text_location_y = y + 20 + (index+1) * 20
                text_location_x = x-self.pivot_img_size
                if text_location_y <= y+h:
                    cv.putText(frame, emotion_label, 
                                (text_location_x, text_location_y), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv.rectangle(frame
                                , (x-self.pivot_img_size+70, y + 13 + (index+1) * 20)
                                , (x-self.pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                , (255,255,255), cv.FILLED)
        return frame
    
    
    def get_emotions(self,frame,face_cord):
        (x,y,w,h)=face_cord
        cv.rectangle(frame, (x,y), (x+w,y+h), (67,67,67), 1)
        face=frame[y:y+h,x:w+x]
        
        gray=functions.detectFace(face,(48,48),True)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_predictions = self.emotion_model.predict(gray)[0,:]
        sum_of_predictions = emotion_predictions.sum()
        mood_items = []
        for i in range(0, len(emotion_labels)):
            mood_item = []
            emotion_label = emotion_labels[i]
            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
            mood_item.append(emotion_label)
            mood_item.append(emotion_prediction)
            mood_items.append(mood_item)
        emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
        emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
        
        #background of mood box
        #transparency
        frame=self.draw_transparenc(frame,(x,y,w,h),self.resx)
        frame=self.draw_emotions(frame,emotion_df,(x,y,w,h),self.resx) 
        return frame,emotion_df.iloc[0]
        
    
    def get_age(self,face_224):						
        age_predictions = self.age_model.predict(face_224)[0,:]
        apparent_age = Age.findApparentAge(age_predictions)
        return int(apparent_age)
        		

    def draw_cont(self,freeze_img,analysis_report,face_cord):
        (x,y,w,h)=face_cord
        info_box_color = (46,200,255)
							
 		#top
        if y - self.pivot_img_size + int(self.pivot_img_size/5) > 0:
            triangle_coordinates = np.array( [
                (x+int(w/2), y)
 				, (x+int(w/2)-int(w/10), y-int(self.pivot_img_size/3))
				, (x+int(w/2)+int(w/10), y-int(self.pivot_img_size/3))
    				] )
								
            cv.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)
            cv.rectangle(freeze_img, (x+int(w/5), y-self.pivot_img_size+int(self.pivot_img_size/5)), 
                         (x+w-int(w/5), y-int(self.pivot_img_size/3)), info_box_color, cv.FILLED)
            cv.putText(freeze_img, analysis_report, (x+int(w/3.5), 
                                                     y - int(self.pivot_img_size/2.1)), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
							
		#bottom
        elif y + h + self.pivot_img_size - int(self.pivot_img_size/5) < self.resy:
            triangle_coordinates = np.array( [
                (x+int(w/2), y+h)
                , (x+int(w/2)-int(w/10), y+h+int(self.pivot_img_size/3))
                , (x+int(w/2)+int(w/10), y+h+int(self.pivot_img_size/3))
    				] )
								
            cv.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)
            cv.rectangle(freeze_img, (x+int(w/5), y + h + int(self.pivot_img_size/3)), 
                          (x+w-int(w/5), y+h+self.pivot_img_size-int(self.pivot_img_size/5)), 
                          info_box_color, cv.FILLED)
            cv.putText(freeze_img, analysis_report, (x+int(w/3.5), 
                                                      y + h + int(self.pivot_img_size/1.5)), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

        return freeze_img			
    
    def get_biggest_face(self,frame):
        self.resx,self.resy=frame.shape[1],frame.shape[0]
        faces=faces=self.face_cascade.detectMultiScale(frame,1.1,3,5)
        if len(faces)==0:
            return None,None
        w_big=0
        indx=0
        for i in range(len(faces)):
            (x,y,w,h)=faces[i]
            if w<150:
                continue
            if w>w_big:
                w_big=w
                indx=i
        face_cord= faces[indx]
        (x,y,w,h)=face_cord
        face=frame[y:y+h,x:w+x] 
        
        face_224=None
        if self.g or self.a:
            face_224 = functions.detectFace(face, (224, 224), False)
        return face_cord,face_224
    
    
    def get_gender(self,face_224):
        gender_prediction = self.gender_model.predict(face_224)[0,:]
        if np.argmax(gender_prediction) == 0:
            gender = "W"
        elif np.argmax(gender_prediction) == 1:
            gender = "M"
        return gender
    
    
    def process(self,_frame,emotions=False,age=False,gender=False,facial_recognition=False):
        frame=_frame.copy()
        face_cord,face_224=self.get_biggest_face(frame)
        if face_cord is None:
            return None
        result={}        
        if emotions:
            frame,emotion=self.get_emotions(frame,face_cord)
            result['e_frame']=frame
            result['emotion']=emotion
        if age:
            age=self.get_age(face_224)
            frame=self.draw_cont(frame,' '+str(age),face_cord)
            result['a_frame']=frame
            result['age']=age
        if gender:
            gender=self.get_gender(face_224)
            frame=self.draw_cont(frame,str(gender),face_cord)
            result['g_frame']=frame
            result['gender']=gender
        if facial_recognition:
            pass
        
        return result
            