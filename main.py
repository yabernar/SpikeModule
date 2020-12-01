import cv2
import numpy as np
from dv import AedatFile
from Frame import Frame

with AedatFile("Captures/circle_travel.aedat4") as f:
    # list all the names of streams in the file
    print(f.names)

    # Access dimensions of the event stream
    height, width = f['events'].size
    list = np.hstack([packet for packet in f['events'].numpy()])
    # frames = Frame(list, (height, width))
    # frames.display()


    # loop through the "events" stream
    for e in f['events']:
        print(e)
        # print(e.x, e.y, e.polarity)

    # End :     1599738641241188
    # Start :   1599738557481268
    # Diff:             83759920
    # 83 s 759 ms 920 us

    # loop through the "frames" stream
    # for frame in f['frames']:
    #     print(frame.timestamp)
    #     cv2.imshow('out', frame.image)
    #     cv2.waitKey(1)
