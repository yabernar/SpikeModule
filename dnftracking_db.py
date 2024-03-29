import os

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, transforms
from matplotlib.transforms import Affine2D

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
stimulus = np.zeros(resolution)
mask = np.ones(resolution, dtype=int)

print(events.shape[0])

def denoising_mask():
    sum_of_events = np.zeros(resolution, dtype=int)
    idx_sum = current_index
    end = min(events.shape[0] - 1, 200000)
    while idx_sum < end:
        sum_of_events[events[idx_sum][1], events[idx_sum][2]] += 1
        idx_sum += 1
    dead_indexes = np.nonzero(sum_of_events >= 100)
    for i in range(len(dead_indexes[0])):
        mask[dead_indexes[0][i], dead_indexes[1][i]] = 0

def update_stimulus():
    global current_index, dnf, start_time, dsec, display, stimulus
    current_end = start_time + timestep
    if current_end//100000 > dsec+dsec_offset:
        dsec += 1
        print(dsec/10)
    start_time = current_end
    dnfx.input.fill(0)
    dnfy.input.fill(0)
    display.fill(0)
    stimulus.fill(0)
    while current_index < events.shape[0] - 1 and events[current_index][0] < current_end:
        stimulus[events[current_index][1], events[current_index][2]] += 0.2
        if events[current_index][3] == 1: display[events[current_index][2], events[current_index][1], 1] = 1
        else: display[events[current_index][2], events[current_index][1], 0] = 1
        current_index += 1
        #dnfx.input[events[current_index][1]] += 0.1
        #dnfy.input[events[current_index][2]] += 0.1
    stimulus *= mask
    mask_display = np.swapaxes(mask, 0, 1)
    mask_display = np.stack((mask_display, mask_display, mask_display), -1)
    display *= mask_display
    dnfx.input = np.sum(stimulus, axis=1)
    dnfy.input = np.sum(stimulus, axis=0)
    #dnfx.input /= np.max(dnfx.input)
    #dnfy.input /= np.max(dnfy.input)

def updatefig(*args):
    update_stimulus()
    dnfx.update_map()
    dnfy.update_map()

    bubble = np.outer(dnfy.activations, dnfx.activations)
    display[:,:,2] = bubble

    inpx.set_ydata(dnfx.input)
    potx.set_ydata(dnfx.potentials)

    inpy.set_ydata(dnfy.input)
    poty.set_ydata(dnfy.potentials)

    img.set_data(display)
    return potx, inpx, inpy, poty, img


if __name__ == '__main__':
    denoising_mask()

    timestep = 10000 # in microseconds
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(11, 8)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 1:])
    dnfx = DNF1D(resolution[0])
    inpx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    potx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    ax1.set_ylim(-3, 10)
    ax1.set_title("X axis")

    r = Affine2D().rotate_deg(90)

    ax2 = fig.add_subplot(gs[1:, 0])
    dnfy = DNF1D(resolution[1])
    inpy, = ax2.plot(list(range(dnfy.width)), dnfy.input)
    inpy.set_transform(r + inpy.get_transform())
    #inpy._transOffset = r + inpy.get_offset_transform()
    poty, = ax2.plot(list(range(dnfy.width)), dnfy.input)
    poty.set_transform(r + poty.get_transform())
    #poty._transOffset = r + inpy.get_offset_transform()
    ax2.set_xlim(-10, 3)
    ax2.set_ylim(0, dnfy.width)
    ax2.invert_yaxis()
    ax2.set_title("Y axis")

    ax3 = fig.add_subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    img = ax3.imshow(display)
    ax3.set_title("Spikes")

    #ax3.get_shared_x_axes().join(ax1, ax2)

    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//1000, blit=True) #//50
    plt.show()
