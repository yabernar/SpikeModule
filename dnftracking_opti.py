import os

import cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, transforms
from matplotlib.transforms import Affine2D

from dnf1d import *
from dnf2d import DNF2D

matplotlib.use("Qt5Agg")
print(matplotlib.get_backend())
np.set_printoptions(threshold=np.inf)

scenario = "highway3"
events = []

with open(os.path.join("baseline", scenario, "events"), 'r') as file:
    data = file.read().splitlines(True)
    for line in data:
        e = line.strip("\n").split("/")
        events.append([int(e[3]), int(e[0]), int(e[1]), int(e[2])])
    events = np.asarray(events)

timescale = 1
start_time = 223015487  # events[0][0]#+3000000
dsec = 0
dsec_offset = events[0][0]//100000
current_index = 0
resolution = (640, 480)
display = np.zeros((480, 640, 3), dtype=float)
stimulus = np.zeros(resolution)
mask = np.ones(resolution, dtype=int)

while current_index < events.shape[0] - 1 and events[current_index][0] < start_time:
    current_index += 1

#image_offset = 350000  # Plane1
#image_offset = 2000000  # walk
image_offset = 0  # highway

print(events.shape[0])

lst_frames = os.listdir(os.path.join("baseline", scenario, "thresholded"))
lst_frames.sort()
print(lst_frames)
current_frame = 0
next_frame_tmstmp = int(lst_frames[1].strip(".jpg"))
current_img = cv2.imread(os.path.join("baseline", scenario, "frames", lst_frames[current_frame]))
saliency = cv2.imread(os.path.join("baseline", scenario, "thresholded", lst_frames[current_frame]))
new_saliency = saliency
inverted = saliency
#saliency = cv2.absdiff(bkg, frame)
#saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
#saliency = cv2.normalize(saliency)
#saliency = cv2.resize(saliency, resolution)

spike_mask = np.zeros((640//20, 480//20), dtype=float)
total_count = 0
missed_count = 0

evaluated_count = 0
evaluated_total = 0

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

def mask_image_generation(bubble, saliency):
    global spike_mask, total_count, missed_count, new_saliency, inverted, evaluated_count, evaluated_total
    spike_mask = cv2.resize(bubble, (640//20, 480//20))
    spike_mask = cv2.threshold(spike_mask, 0.20, 1, cv2.THRESH_BINARY)[1]
    inverted = cv2.threshold(spike_mask, 0.20, 1, cv2.THRESH_BINARY_INV)[1]
    evaluated_count += np.count_nonzero(spike_mask)
    evaluated_total += np.prod(spike_mask.shape)
    spike_mask = np.kron(spike_mask, np.ones((20, 20)))
    inverted = np.kron(inverted, np.ones((20, 20)))
    new_saliency = saliency[:,:,0]
    total_count += np.count_nonzero(new_saliency)
    missed_count += np.count_nonzero(np.multiply(inverted, new_saliency))
    print("Total :", total_count, "Missed :", missed_count)
    print("Total :", evaluated_total, "Evaluated :", evaluated_count)

def update_stimulus():
    global current_index, dnf, start_time, dsec, display, stimulus, saliency, next_frame_tmstmp, current_frame, current_img
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
    if current_frame < len(lst_frames) - 2 and next_frame_tmstmp + image_offset <= current_end:
        current_frame += 1
        next_frame_tmstmp = int(lst_frames[current_frame+1].strip(".jpg"))
        saliency = cv2.imread(os.path.join("baseline", scenario, "thresholded", lst_frames[current_frame]))
        current_img = cv2.imread(os.path.join("baseline", scenario, "frames", lst_frames[current_frame]))
        current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)

        bubble = np.outer(dnfx.activations, dnfy.activations)
        bubble = np.rot90(bubble)
        bubble = np.flipud(bubble)
        mask_image_generation(bubble, saliency)
        #saliency = cv2.absdiff(bkg, frame)
        #saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
        #saliency = cv2.normalize(saliency)
        #saliency = cv2.resize(saliency, resolution)


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
    frame.set_data(saliency)
    msk.set_data(spike_mask)
    tmp.set_data(current_img)

    return potx, inpx, inpy, poty, img, frame, msk, tmp

if __name__ == '__main__':
    denoising_mask()

    timestep = 10000 # in microseconds
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(11, 8)
    gs = fig.add_gridspec(7, 7)

    ax1 = fig.add_subplot(gs[0, 1:4])
    dnfx = DNF1D(resolution[0])
    inpx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    potx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    ax1.set_ylim(-3, 10)
    ax1.set_title("X axis")

    r = Affine2D().rotate_deg(90)

    ax2 = fig.add_subplot(gs[1:4, 0])
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

    ax3 = fig.add_subplot(gs[1:4, 1:4], sharex=ax1, sharey=ax2)
    img = ax3.imshow(display)
    ax3.set_title("Spikes")

    #ax3.get_shared_x_axes().join(ax1, ax2)
    ax4 = fig.add_subplot(gs[1:4, 4:7])
    frame = ax4.imshow(saliency)
    ax4.set_title("Saliency")

    ax5 = fig.add_subplot(gs[4:7, 1:4])
    msk = plt.imshow(saliency, vmin=0, vmax=1) #, cmap='hot', interpolation='nearest', animated=True)
    ax5.set_title("Spike mask")

    ax5 = fig.add_subplot(gs[4:7, 4:7])
    tmp = plt.imshow(current_img) #, cmap='hot', interpolation='nearest', animated=True)
    ax5.set_title("Recording")

    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//1000, blit=True) #//50
    plt.show()
