import threading 
import os

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def f():
    # print('gggg gg')
    duration = 1
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    # print('ffffff')
    
r=False
p=StoppableThread(target=f)
i=0
while True:
    # print(i)
    i+=1
    if i%100000==0:
        r=True
    if not p.isAlive() and r:
        print('here')
        p=None
        p=StoppableThread(target=f)
        p.start()
        # p.stop()
        print('there')
        r=False

# print(p.isAlive())
# p.start()
# print(p.isAlive())
# p.join()
# p.terminate()
# print(p.is_alive())
# # print(p.is_alive())