import dlib
import cv2
import time
import numpy as np
import torch
from gazetracker.ehg_nfeat24_B2_Hg1_D4_E1 import HourglassNet
import os
import multiprocessing as mltp

class Gaze:
    # -----------global varaibles-----------
    predictor_path = 'gazetracker/shape_predictor_68_face_landmarks.dat'
    distracted_limit=distracted_counter=20
    last_frame_index = 0
    last_frame_time = time.time()
    fps_history = []
    all_gaze_histories = []
    
    _last_frame_index = 0
    _indices = []
    _frames = {}
    
    batch_size = 2
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        print(os.path.isdir(self.predictor_path))     
        self.predictor = dlib.shape_predictor(self.predictor_path)
        model =  HourglassNet()
        self.model = torch.nn.DataParallel(model)
        checkpoint = torch.load('gazetracker/ehg_nfeat24_B2_Hg1_D4_E1.pth.tar', map_location='cpu')
        checkpoint['state_dict'] = {k:v for (k, v) in checkpoint['state_dict'].items() if k.split('_')[-1] != 'tracked'}  
        self.model.load_state_dict(checkpoint['state_dict'])
        self.warning_p=mltp.Process(target=self.warning)
        print('Eye initaillized successfully')
    
    
    def warning(self):
        duration = 0.01
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    
    def detect_face(self,frame):
        frame_index = frame['frame_index']
        previous_index = self._indices[self._indices.index(frame_index) - 1]
        previous_frame = self._frames[previous_index]
        # resize and detect
        if ('last_face_detect_index' not in previous_frame 
            or frame['frame_index'] - previous_frame['last_face_detect_index'] > 59):
            faces = []
            rects = self.detector(cv2.resize(frame['gray'], (0, 0), fx=0.5, fy=0.5), 0)            
            # If no output to visualize, show unannotated frame
            if len(rects) == 0:
                frame['faces'] = None
                return frame
            for d in rects:
                l, t, r, b = d.left(), d.top(), d.right(), d.bottom()
                l *= 2
                t *= 2
                r *= 2
                b *= 2
                w, h = r - l, b - t
            faces.append((l, t, w, h))
            faces.sort(key=lambda bbox: bbox[0])
            frame['faces'] = faces
            frame['last_face_detect_index'] = frame['frame_index']
            previous_frame['landmarks'] = []
    
        else:
            frame['faces'] = previous_frame['faces']
            frame['last_face_detect_index'] = previous_frame['last_face_detect_index']
        return frame
        
    
    def crop_eye(self,frame):
        '''
        Return:
            list[Dict(image: eye image, inv transform: matrix, side: str),  ...]
        '''
    
        eyes = []
        # Final output dimensions
        oh, ow = 48, 64
    
        # Select which landmarks (raw/smoothed) to use
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])
    
        for face, landmarks in zip(frame['faces'], frame_landmarks):
            # Segment eyes
             i = 0
             for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
    #            for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
                x1, y1 = landmarks[corner1, :]
                x2, y2 = landmarks[corner2, :]
                eye_width = 2 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
                if eye_width == 0.0:
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                # Centre image on middle of eye
                translate_mat = np.asmatrix(np.eye(3))
                translate_mat[:2, 2] = [[-cx], [-cy]]
                inv_translate_mat = np.asmatrix(np.eye(3))
                inv_translate_mat[:2, 2] = -translate_mat[:2, 2]
    
                # Rotate to be upright
                roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
                rotate_mat = np.asmatrix(np.eye(3))
                cos = np.cos(-roll)
                sin = np.sin(-roll)
                rotate_mat[0, 0] = cos
                rotate_mat[0, 1] = -sin
                rotate_mat[1, 0] = sin
                rotate_mat[1, 1] = cos
                inv_rotate_mat = rotate_mat.T
    
                # Scale
                scale = ow / eye_width
                scale_mat = np.asmatrix(np.eye(3))
                scale_mat[0, 0] = scale_mat[1, 1] = scale
                inv_scale = 1.0 / scale
                inv_scale_mat = np.asmatrix(np.eye(3))
                inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale
    
                # Centre image
                centre_mat = np.asmatrix(np.eye(3))
                centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
                inv_centre_mat = np.asmatrix(np.eye(3))
                inv_centre_mat[:2, 2] = -centre_mat[:2, 2]
    
                # Get rotated and scaled, and segmented image
                transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
                inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                     inv_centre_mat)
                eye_image = cv2.warpAffine(frame['gray'], transform_mat[:2, :], (ow, oh))
                # process for hourglass
                eye_image = cv2.equalizeHist(eye_image)
                eye_image = eye_image.astype(np.float32)
                eye_image /= 255.0
                if is_left:
                    eye_image = np.fliplr(eye_image)
                eyes.append({'image': eye_image,
                             'inv_landmarks_transform_mat': inv_transform_mat,
                             'side': 'left' if is_left else 'right',
                             'eye_index': i})
                i += 1
        frame['eyes'] = eyes
    
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])
        return frame
    
    
    def find_landmarks(self,heatmap):
        '''
        Args:
            heatmap:
                numpy([2, C, H, W])
        Return:
            preds:
                tensor([1, 18, 2])
        '''
        masks = np.zeros((2, 18, 2))
        for i, hms in enumerate(heatmap):
            for j, hm in enumerate(hms):            
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(hm)            
                masks[i, j] = maxLoc  
                
        return masks

    
    def is_distracted(self,yaw,pitch):
        if yaw>20 or yaw<-20 or pitch>20 or pitch<-20:
            self.distracted_counter-=1
        else:
            self.distracted_counter=self.distracted_limit
            
        if self.distracted_counter<=0:
            return True
        return False
    
    
    def draw_gaze(self,image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])
        cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                       tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                       thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out
    
    
    
    def process(self,rgb):
        start = time.time()
        before_frame_read = time.time()
        next_frame_index = self.last_frame_index + 1
        current_index = self._last_frame_index + 1
        after_frame_read = time.time()
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        frame = {'frame_index': current_index,
                 'time': {'before_frame_read': before_frame_read,
                          'after_frame_read': after_frame_read,},
                 'bgr': rgb,
                 'gray': gray,}

        self._frames[current_index] = frame
        self._indices.append(current_index)
        
        ##############
        #Face Detector
        ##############
        frame = self.detect_face(frame)
        if frame['faces'] == None:
            return rgb,False
        
        ##############  
        #Face Landmarks
        ##############    
        landmarks = []
        for face in frame['faces']:
            l, t, w, h = face
            rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
            landmarks_dlib = self.predictor(frame['gray'], rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts
            landmarks.append(np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)]))
        frame['landmarks'] = landmarks
        
        ##############
        #Eye Detector
        ############## 
        frame = self.crop_eye(frame)
        
        ##############
        # CNN
        ##############
        # Feed eye into NN [2, 1, H, W]
        eye_left = np.expand_dims(frame['eyes'][0]['image'], 0)
        eye_right = np.expand_dims(frame['eyes'][1]['image'], 0)
        np_input = np.zeros([2, 1, eye_right.shape[1], eye_right.shape[2]])
        np_input[0] = eye_left
        np_input[1] = eye_right
        torch_input = torch.from_numpy(np_input).float()
        timenow = time.time()
        output = self.model(torch_input) # [b, 18, H, W]
        heatmaps = output[-1].detach().cpu().numpy()
        
        ##############
        # Eye Landmarks
        ##############         
        landmarks = self.find_landmarks(heatmaps)

        bgr = None
        for j in range(self.batch_size):
            start_time = time.time()
            eye_index = frame['eyes'][j]['eye_index']
            bgr = frame['bgr']
            eye = frame['eyes'][j]
            eye_image = eye['image']
            eye_side = eye['side']
            eye_landmarks = landmarks[j]
            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                eye_image = np.fliplr(eye_image)
                
            #######################
            # Embedded eye
            #######################
            # Embed eye image and annotate for picture-in-picture
            eye_upscale = 2
            eye_image = (eye_image - eye_image.min()) * (1/(eye_image.max() - eye_image.min()) * 255)
            eye_image = eye_image.astype(np.uint8)
            eye_image_raw = cv2.cvtColor(cv2.equalizeHist(eye_image), cv2.COLOR_GRAY2BGR)
            eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
            eye_image_annotated = np.copy(eye_image_raw)
            
            # Drawings
            cv2.polylines(eye_image_annotated,
                         [np.round(eye_upscale*eye_landmarks[0:16]).astype(np.int32).reshape(-1, 1, 2)],
                         isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA,)
            
            cv2.drawMarker(eye_image_annotated,
                          tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                          color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                          thickness=1, line_type=cv2.LINE_AA,)
            
            # store it in bgr frame
            face_index = int(eye_index / 2)
            eh, ew, _ = eye_image_raw.shape
            v0 = face_index * 2 * eh
            v1 = v0 + eh
            v2 = v1 + eh
            u0 = 0 if eye_side == 'left' else ew
            u1 = u0 + ew
            bgr[v0:v1, u0:u1] = eye_image_raw
            bgr[v1:v2, u0:u1] = eye_image_annotated 
            
            # Transform predictions
            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                               'constant', constant_values=1.0))
            eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
            eye_landmarks = np.asarray(eye_landmarks)
            eyelid_landmarks = eye_landmarks[0:8, :]
            iris_landmarks = eye_landmarks[8:16, :]
            iris_centre = eye_landmarks[16, :]
            eyeball_centre = eye_landmarks[17, :]
#            eyeball_radius = np.linalg.norm(eye_landmarks[10, :]-eye_landmarks[9, :])
            eyeball_radius = 22.5
            
            # Drawing
            for f, face in enumerate(frame['faces']):
                cv2.drawMarker(bgr,
                              tuple(np.round(iris_centre).astype(np.int32)),
                              color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                              thickness=1, line_type=cv2.LINE_AA,)
                for i, landmark in enumerate(eye_landmarks):
                    # landmark
                    cv2.circle(bgr, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
                    
            ####################################
            # Gaze stuff
            ####################################
            # Smooth and visualize gaze direction
            num_total_eyes_in_frame = len(frame['eyes'])
            if len(self.all_gaze_histories) != num_total_eyes_in_frame:
                self.all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
            gaze_history = self.all_gaze_histories[eye_index]

            # Draw gaze
            pitch = -np.arcsin(np.clip((iris_centre[1] - eyeball_centre[1]) / eyeball_radius, -1.0, 1.0))
            yaw = np.arcsin(np.clip((iris_centre[0] - eyeball_centre[0]) / (eyeball_radius * -np.cos(pitch)),
                                    -1.0, 1.0))
            current_gaze = np.array([pitch, yaw])
            
            pitch*=100
            yaw*=100
            # uncomment the following lines for drawing the direction of gaze
            # yaw_text=pitch_text=dist_text=''
            # if yaw>20:
            #     yaw_text='Right'
            # elif yaw<-20:
            #     yaw_text='Left'
            
            # if pitch>20:
            #     pitch_text='Up'
            # elif pitch<-20:
            #     pitch_text='Down'
            
            if self.is_distracted(yaw,pitch):
                self.warning()
        
        
            # uncomment the following lines for drawing the distraction label
            # # global distracted_count
            # cv2.putText(bgr,dist_text,org=(400,100),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
            #                color=(235, 25, 23), thickness=5, lineType=cv2.LINE_AA)
            
            # cv2.putText(bgr,yaw_text,org=(100,400),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
            #                color=(235, 25, 23), thickness=5, lineType=cv2.LINE_AA)
            # cv2.putText(bgr,pitch_text,org=(100,440),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
            #                color=(235, 25, 23), thickness=5, lineType=cv2.LINE_AA)
            
            gaze_history.append(current_gaze)
            gaze_history_max_len = 10
            if len(gaze_history) > gaze_history_max_len:
                gaze_history = gaze_history[-gaze_history_max_len:]
            mean_gaze = np.mean(gaze_history, axis=0)
            self.draw_gaze(bgr, iris_centre, mean_gaze,
                                length=100.0, thickness=1)
            if eye_index == len(frame['eyes']) - 1:
                # Calculate timings
                fh, fw, _ = bgr.shape
                cv2.putText(bgr,  'FPS: ' + str(1/(time.time()-start)), org=(fw - 151, fh - 21),
                           fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.50,
                           color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
                
        return rgb