from multiprocessing import Process, Queue
import cv2
import numpy as np
from posestimator.mark_detector import MarkDetector
from posestimator.os_detector import detect_os
from posestimator.pose_estimator import PoseEstimator
from posestimator.stabilizer import Stabilizer

class Head:
    CNN_INPUT_SIZE = 128
    img_queue=box_queue=pose_stabilizers=mark_detector=pose_estimator=None
    box_process=None
    
    def __init__(self,sample_frame):
        detect_os()
        self.mark_detector = MarkDetector()

        # Setup process and queues for multiprocessing.
        self.img_queue = Queue()
        self.box_queue = Queue()
        self.img_queue.put(sample_frame)
        box_process = Process(target=self.get_face, args=(
            self.mark_detector, self.img_queue, self.box_queue,))
        box_process.start()
        height, width = sample_frame.shape[:2]
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        self.pose_estimator = PoseEstimator(img_size=(height, width))
    
        # Introduce scalar stabilizers for pose.
        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]
        
        
    def get_face(self,detector, img_queue, box_queue):
        """Get face from image queue. This function is used for multiprocessing"""
        while True:
            image = img_queue.get()
            box = detector.extract_cnn_facebox(image)
            box_queue.put(box)
    
    def process(self,frame):
        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]
    
        # If frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2)
    
        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose
    
        # Feed frame to image queue.
        self.img_queue.put(frame)
    
        # Get face from box queue.
        facebox = self.box_queue.get()
    
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (self.CNN_INPUT_SIZE, 
                                             self.CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
            marks = self.mark_detector.detect_marks([face_img])
    
            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
    
            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))
    
            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])
    
            # Try pose estimation with 68 points.
            pose = self.pose_estimator.solve_pose_by_68_points(marks)
    
            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))
    
            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))
    
            # Uncomment following line to draw stabile pose annotation on frame.
            rotation=self.pose_estimator.draw_annotation_box(
                frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
    
            # Uncomment following line to draw head axes on frame.
            # pose_estimator.draw_axes(frame, stabile_pose[0], stabile_pose[1])
            return frame,rotation
        return frame,None
        
        def __del__(self):
            # Clean up the multiprocessing process.
            self.box_process.terminate()
            self.box_process.join()
