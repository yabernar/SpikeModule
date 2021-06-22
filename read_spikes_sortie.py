import cv2
import numpy as np
from Frame import Frame

with open("Sortie/Plane2/camera1/spikes", 'r') as f:

    # Use txt file for events
    height, width = (480, 640)
    spikes = []
    for line in f:
        split= line.strip('\n').split('/')
        spikes.append([int(split[3]), int(split[0]), int(split[1])])
    spikes = np.asarray(spikes)
    print(spikes.shape)

    frames = Frame(spikes, (height, width))
    frames.display()