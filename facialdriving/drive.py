from facialdriving.new_env import CarlaEnv

class Driver:
    def __init__(self):
        self.env=CarlaEnv()
        self.env.reset()
        self.speed=0.5
        self.brake=0

    def process(self,rotation):
        diff=45-int(rotation)
        if diff>0:
            diff=-0.1
        if diff<0:
            diff=0.1
        self.env.step((diff,
                       self.speed,
                       self.brake))
