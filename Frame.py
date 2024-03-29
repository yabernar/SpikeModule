import os

import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)


class Frame:
    def __init__(self, events, size):
        self.fps = 60
        self.offset = 0
        self.events = events
        self.time = self.events[0][0]
        self.current_index = 0
        self.frame = np.zeros(size, dtype=int)

    def display(self, path="Results/Test"):
        nb = 0
        while self.current_index < self.events.shape[0]-1:
            current_end = self.time + (1 / self.fps) * 1000000  # In microseconds
            self.frame.fill(0)
            while self.current_index < self.events.shape[0]-1 and self.events[self.current_index][0] < current_end:
                self.frame[self.events[self.current_index][2], self.events[self.current_index][1]] = 255
                self.current_index += 1
            self.time = current_end
            if nb > self.offset:
                img = Image.fromarray(np.uint8(self.frame))
                img.save(os.path.join(path, str(int(self.time))+".png"))
                #img.save(os.path.join(path, "frame"+str(nb)+".png"))
            nb += 1