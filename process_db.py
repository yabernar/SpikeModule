import os
import shutil

import cv2
import numpy as np
from Frame import Frame

inp = "Sortie"
out = "baseline"
offset = 900 * 16667
dirs = os.listdir(inp)
dirs = ["Plane1"]

for folder in dirs:
    print("Starting", folder, end="... ")
    out_path = os.path.join(out, folder.lower())
    in_path = os.path.join(inp, folder)
    os.makedirs(out_path, exist_ok=True)

    # Copying Spikes
    print("Copying events", end="... ")
    file = os.listdir(os.path.join(in_path, "camera1"))
    shutil.copyfile(os.path.join(in_path, "camera1", file[0]), os.path.join(out_path, "events"))
    with open(os.path.join(out_path, "events"), 'r') as fin:
        data = fin.read().splitlines(True)
        f_frame = data[5]
        if offset > 0:
            start = 8
            end = len(data)-1
            target = int(data[start].strip('\n').split('/')[3]) + offset
            while (end - start) > 1:
                middle = int((start + end)/2)
                if int(data[middle].strip('\n').split('/')[3]) < target:
                    start = middle
                else:
                    end = middle
            cut = start
        else:
            cut = 8
    with open(os.path.join(out_path, "events"), 'w') as fout:
        fout.writelines(data[cut:])

    # Copying Frames
    print("Copying frames", end="... ")

    # Reading first frame timestamp
    f_hour, f_minute, f_seconds, f_microsecs = int(f_frame[37:39]), int(f_frame[40:42]), int(f_frame[43:45]), int(f_frame[46:53])
    f_timestamp = int(f_frame[18:27])

    os.makedirs(os.path.join(out_path, "frames"), exist_ok=True)
    lst_frames = os.listdir(os.path.join(in_path, "frames"))
    lst_frames.sort()
    conversion = {}
    for frame in lst_frames:
        hours, minutes, seconds, microsecs = int(frame[6:8]), int(frame[9:11]), int(frame[12:14]), int(frame[15:21])
        elapsed_time = (((hours - f_hour) * 60 + minutes - f_minute) * 60 + seconds - f_seconds) * 1000000 + microsecs - f_microsecs
        if elapsed_time > offset:
            conversion[frame] = f_timestamp + elapsed_time
    for frame in conversion:
        shutil.copyfile(os.path.join(in_path, "frames", frame), os.path.join(out_path, "frames", str(conversion[frame])+".jpg"))
#    if not os.path.exists(os.path.join(out_path, "frames")):
#        shutil.copytree(os.path.join(in_path, "frames"), os.path.join(out_path, "frames"))

    # Generating SpikeFrames
    print("Generating Spike-frames", end="... ")
    with open(os.path.join(out_path, "events"), 'r') as f:
        height, width = (480, 640)
        spikes = []
        for line in f:
            split= line.strip('\n').split('/')
            spikes.append([int(split[3]), int(split[0]), int(split[1])])
        spikes = np.asarray(spikes)

        os.makedirs(os.path.join(out_path, "spike_frames_60fps"), exist_ok=True)
        frames = Frame(spikes, (height, width))
        frames.display(path=os.path.join(out_path, "spike_frames_60fps"))

    print("Done !")