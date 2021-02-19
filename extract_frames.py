import os
import cv2
from dv import AedatFile

with AedatFile(os.path.join("Captures", "color_office.aedat4")) as f:
    # list all the names of streams in the file
    print(f.names)

    # Access dimensions of the event stream
    height, width = f['events'].size

    # loop through the "events" stream
    for e in f['events']:
        print(e.timestamp)
        break

    nbr = 0
    # loop through the "frames" stream
    for frame in f['frames']:
        if nbr == 420:
            print(nbr, ":", frame.timestamp)
        cv2.imwrite(os.path.join("Frames", "src", "office"+str(nbr)+".png"), frame.image)
        nbr += 1
        cv2.imshow('out', frame.image)
        cv2.waitKey(1)
