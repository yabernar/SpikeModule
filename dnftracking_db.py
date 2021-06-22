import os

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dnf1d import *

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


def update_stimulus():
    global current_index, dnf, start_time, dsec
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


def updatefig(*args):
    update_stimulus()
    dnfx.update_map()
    dnfy.update_map()

    bubble = np.outer(dnfy.activations, dnfx.activations)
    display[:,:,2] = bubble

    inpx.set_ydata(dnfx.input)
    potx.set_ydata(dnfx.potentials)

    inpy.set_array(dnfy.input)
    poty.set_array(dnfy.potentials)

    img.set_data(display)
    return potx, inpx, inpy, poty, img


if __name__ == '__main__':
    timestep = 10000 # in microseconds
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 1:])
    dnfx = DNF1D(resolution[0])
    inpx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    potx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    ax1.set_ylim(-3, 3)
    ax1.set_title("X axis")

    ax2 = fig.add_subplot(gs[1:, 0])
    dnfy = DNF1D(resolution[1])
    inpy = ax2.hlines(list(range(dnfy.width)), 0, resolution[1])
    poty = ax2.hlines(list(range(dnfy.width)), 0, resolution[1])
    ax2.set_xlim(0, 1)
    ax2.set_title("Y axis")

    ax3 = fig.add_subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    img = ax3.imshow(display)
    ax3.set_title("Spikes")

    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//1000, blit=True) #//50
    plt.show()
