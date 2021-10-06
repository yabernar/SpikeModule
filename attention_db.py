import os

import cv2
import numpy as np
from dv import AedatFile
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dnf1d import *
from dnf2d import DNF2D

matplotlib.use("Qt5Agg")
print(matplotlib.get_backend())
np.set_printoptions(threshold=np.inf)

scenario = "plane1"
events = []

with open(os.path.join("baseline", scenario, "events"), 'r') as file:
    data = file.read().splitlines(True)
    for line in data:
        e = line.strip("\n").split("/")
        events.append([int(e[3]), int(e[0]), int(e[1]), int(e[2])])
    events = np.asarray(events)

timescale = 1
start_time = events[0][0]
dsec = 0
dsec_offset = events[0][0]//100000
current_index = 0
resolution = (640, 480)
display = np.zeros((480, 640, 3), dtype=float)

lst_frames = os.listdir(os.path.join("baseline", scenario, "frames"))
lst_frames.sort()
print(lst_frames)
current_frame = 0
next_frame_tmstmp = int(lst_frames[1].strip(".jpg"))
bkg = cv2.imread(os.path.join("baseline", scenario, "bkg.jpg"))
frame = cv2.imread(os.path.join("baseline", scenario, "frames", lst_frames[current_frame]))
saliency = cv2.absdiff(bkg, frame)
#saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
#saliency = cv2.normalize(saliency)
saliency = cv2.resize(saliency, resolution)

def update_stimulus():
    global current_index, current_frame, dnf, start_time, dsec, saliency, next_frame_tmstmp
    current_end = start_time + timestep
    if current_end//100000 > dsec+dsec_offset:
        dsec += 1
        print(dsec/10)
    start_time = current_end
    dnfx.input.fill(0)
    dnfy.input.fill(0)
    display.fill(0)
    while current_index < events.shape[0] - 1 and events[current_index][0] < current_end:
        dnfx.input[events[current_index][1]] += 0.3
        dnfy.input[events[current_index][2]] += 0.3
        if events[current_index][3] == 1: display[events[current_index][2], events[current_index][1], 1] = 1
        else: display[events[current_index][2], events[current_index][1], 0] = 1
        current_index += 1
    if current_frame < len(lst_frames) - 2 and next_frame_tmstmp <= current_end:
        current_frame += 1
        next_frame_tmstmp = int(lst_frames[current_frame+1].strip(".jpg"))
        frame = cv2.imread(os.path.join("baseline", scenario, "frames", lst_frames[current_frame]))
        saliency = cv2.absdiff(bkg, frame)
        #saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
        #saliency = cv2.normalize(saliency)
        saliency = cv2.resize(saliency, resolution)

def updatefig(*args):
    update_stimulus()
    dnfx.update_map()
    dnfy.update_map()

    bubble = np.outer(dnfy.activations, dnfx.activations)
    display[:,:,2] = bubble

    attention_input = bubble[:saliency.shape[0], :saliency.shape[1]] + saliency[:,:,0]/255
    res = np.zeros((attention_input.shape[0]//10, attention_input.shape[1]//10))
    px = np.vsplit(attention_input, attention_input.shape[0]//10)
    for i in range(len(px)):
        px2 = np.hsplit(px[i], attention_input.shape[1]//10)
        for j in range(len(px2)):
            res[i, j] = np.mean(px2[j])
    dnf_att.input = res
    dnf_att.update_map()

    inpx.set_ydata(dnfx.input)
    potx.set_ydata(dnfx.potentials)

    inpy.set_array(dnfy.input)
    poty.set_array(dnfy.potentials)

    img.set_data(display)
    frame.set_data(saliency)

    att_pot.set_data(dnf_att.potentials)
    att_dec.set_data(dnf_att.activations)
    return potx, inpx, inpy, poty, img, frame, att_pot, att_dec


if __name__ == '__main__':
    timestep = 10000 # in microseconds
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(7, 7)

    ax1 = fig.add_subplot(gs[0, 1:4])
    dnfx = DNF1D(resolution[0])
    inpx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    potx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    ax1.set_ylim(-3, 3)
    ax1.set_title("X axis")

    ax2 = fig.add_subplot(gs[1:4, 0])
    dnfy = DNF1D(resolution[1])
    inpy = ax2.hlines(list(range(dnfy.width)), 0, resolution[1])
    poty = ax2.hlines(list(range(dnfy.width)), 0, resolution[1])
    ax2.set_xlim(0, 1)
    ax2.set_title("Y axis")

    ax3 = fig.add_subplot(gs[1:4, 1:4], sharex=ax1, sharey=ax2)
    img = ax3.imshow(display)
    ax3.set_title("Spikes")

    ax4 = fig.add_subplot(gs[1:4, 4:7])
    frame = ax4.imshow(saliency)
    ax4.set_title("Saliency")

    dnf_att = DNF2D(26, 34)
    ax5 = fig.add_subplot(gs[4:7, 1:4])
    att_pot = plt.imshow(dnf_att.potentials, vmin=-6, vmax=6) #, cmap='hot', interpolation='nearest', animated=True)
    ax5.set_title("Attention Potentials")

    ax5 = fig.add_subplot(gs[4:7, 4:7])
    att_dec = plt.imshow(dnf_att.activations, vmin=0, vmax=1) #, cmap='hot', interpolation='nearest', animated=True)
    ax5.set_title("Attention Decision")

    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//1000, blit=True) #//50
    plt.show()
