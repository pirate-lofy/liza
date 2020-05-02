from facialdriving.carla.client import CarlaClient
from facialdriving.carla.settings import CarlaSettings
from facialdriving.carla.sensor import Camera
from facialdriving.carla.tcp import TCPConnectionError

import random

class CarlaEnv:
    repeat_frames=3

    def __init__(self):
        # settings
        self.settings=CarlaSettings()
        self.settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=0,
            NumberOfPedestrians=0,
            WeatherId=1,
            QualityLevel='Low',
#            ServerTimeOut=10000
        )
        self.settings.randomize_seeds()
        
        # add camera
        cam=Camera('rgb')
        cam.set(FOV=90.0)
        cam.set_image_size(800,600)
        cam.set_position(x=0.8,y=0,z=1.3)
        self.settings.add_sensor(cam)
        
        self._connect()
        self.scene=self.client.load_settings(self.settings)

    def _connect(self):
        while True:
            try:
                self.client=CarlaClient('localhost',2000,10)
                self.client.connect()
                print('CarlaEnv log: client connected')
                break
            except TCPConnectionError:
                print('server not launched yet')
    
    def reset(self):
        print('resetting')
        self._start_new_episod()
        self._empty_cycle()
        self._get_data()
    
    def step(self,actions):
        steer,throttle,brake=actions
        for _ in range(self.repeat_frames):
            self.client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    hand_brake=False,
                    reverse=False                
                    )
            self.client.read_data()
        self.client.send_control(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    hand_brake=False,
                    reverse=False                
                    )
        done=self._get_data()
        if done:
             self.reset()
  
    def _empty_cycle(self):
        print('CarlaEnv log: empty cycle started...')
        for _ in range(30):
            self.client.read_data()
            self.client.send_control(
                steer=0,
                throttle=0,
                brake=0,
                hand_brake=False,
                reverse=False                
                )
        print('CarlaEnv log: empty cycle ended.\n')
    
 


    def _is_done(self,measures):
        pm=measures.player_measurements
        cols=pm.collision_vehicles+pm.collision_other+\
                pm.collision_pedestrians+pm.intersection_offroad
        return cols>0
    
    def _get_data(self):
        measures,_=self.client.read_data()
        done=self._is_done(measures)
        return done


    def _start_new_episod(self):
        n_spots=len(self.scene.player_start_spots)
        start_point=random.randint(0,max(0,n_spots-1))
        self.client.start_episode(start_point)

