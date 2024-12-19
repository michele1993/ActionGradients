import numpy as np


class MemoryBuffer:
    """ Super simple memory buffer to store transition and sample single (not-batch) transitions"""

    def __init__(self,size = 1):
        self.size = size
        self.buffer = []
        self.c_size = 0

    def store_transition(self,c_state,action,rwd, p_action):


        # Check if the replay buffer is full
        if len(self.buffer) < self.size:
            self.buffer.append((c_state,action,rwd,p_action))

        # if full, start replacing values from the first element
        else:

            self.buffer[self.c_size] = (c_state,action,rwd,p_action)
            self.c_size+=1

            # Need to restart indx when reach end of list
            if self.c_size == self.size:
                self.c_size = 0

    def sample_transition(self):
        rand_indx = np.random.randint(len(self.buffer))
        return self.buffer[rand_indx]
