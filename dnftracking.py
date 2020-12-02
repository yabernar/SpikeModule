import numpy as np
from dv import AedatFile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dnf1d import *


np.set_printoptions(threshold=np.inf)

with AedatFile("Captures/drone_top_fisheye_fast.aedat4") as f:
    # Access dimensions of the event stream
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])
timescale = 50
offset = 11000000
start_time = events[0][0] + offset
dsec = offset//100000
dsec_offset = events[0][0]//100000
current_index = 0
# Removing events before the offset
while current_index < events.shape[0] - 1 and events[current_index][0] < start_time:
    current_index += 1

# Simulation parameters
resolution = (640, 480)

def update_stimulus():
    global current_index, dnf, start_time, dsec
    current_end = start_time + timestep
    if current_end//100000 > dsec+dsec_offset:
        dsec += 1
        print(dsec/10)
    start_time = current_end
    dnfx.input.fill(0)
    dnfy.input.fill(0)
    while current_index < events.shape[0] - 1 and events[current_index][0] < current_end:
        dnfx.input[events[current_index][1]] += 0.3
        dnfy.input[events[current_index][2]] += 0.3
        current_index += 1

def updatefig(*args):
    update_stimulus()
    dnfx.update_map()
    dnfy.update_map()

    inpx.set_ydata(dnfx.input)
    potx.set_ydata(dnfx.potentials)

    inpy.set_ydata(dnfy.input)
    poty.set_ydata(dnfy.potentials)

    return potx, inpx, inpy, poty

if __name__ == '__main__':
    timestep = 1000 # in microseconds
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    dnfx = DNF1D(resolution[0])
    inpx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    potx, = ax1.plot(list(range(dnfx.width)), dnfx.input)
    ax1.set_ylim(-3, 3)
    ax1.set_title("X axis")

    dnfy = DNF1D(resolution[1])
    inpy, = ax2.plot(list(range(dnfy.width)), dnfy.input)
    poty, = ax2.plot(list(range(dnfy.width)), dnfy.input)
    ax2.set_ylim(-3, 3)
    ax2.set_title("Y axis")
    
    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//200, blit=True)
    plt.show()
