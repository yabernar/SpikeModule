from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from dv import AedatFile
from PIL import Image

np.set_printoptions(threshold=np.inf)

with AedatFile("Captures/circle_travel.aedat4") as f:
    # Access dimensions of the event stream
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])
timestep = 200
offset = 1800000
start_time = events[0][0] + offset
dsec = offset//100000
dsec_offset = events[0][0]//100000
current_index = 0
# Removing events before the offset
while current_index < events.shape[0] - 1 and events[current_index][0] < start_time:
    current_index += 1

# Simulation parameters
resolution = (640, 480)
tau = 500*us
El = -0.7

# LIF
eqs_LIF = '''dv/dt = ((El - v) + I)/tau : 1
        I : 1'''
# Not LIF
eqs_2 = '''dv/dt = (1-v)/tau : 1'''

# Create input Layer (DO NOT USE SELF WITH BRIAN OBJECTS)
G = NeuronGroup(np.prod(resolution), eqs_LIF, threshold='v > 0.9', reset='v = 0', method='linear')
G.v = 0
Spikes = SpikeMonitor(G)
M = StateMonitor(G, 'v', record=True)

# Create processing Layer

# Connecting two layers with synapses


@network_operation(dt=timestep * us)
def update_stimulus(t):
    global current_index, G, start_time, dsec
    tick = int(t.variable.get_value()*1000000)  # time as us
    current_end = start_time + tick
    if current_end//100000 > dsec+dsec_offset:
        dsec += 1
        print(dsec/10)
    start_time = current_end
    G.I = 0
    while current_index < events.shape[0] - 1 and events[current_index][0] < current_end:
        G[(resolution[1] - events[current_index][2] - 1) * resolution[0] + events[current_index][1]].I += 1
        current_index += 1


spike_count = 0
def updatefig(*args):
    global spike_count
    run(timestep*us)
    x, y = Spikes.i[:]%resolution[0], Spikes.i[:]//resolution[0]
    data = np.column_stack((x[spike_count:], y[spike_count:]))
    spike_count = len(x)
    spk.set_offsets(data)
    return spk,


if __name__ == '__main__':
    # timestep = 200 # in microseconds defined above
    fig, ax = plt.subplots()

    spk = ax.scatter([], [], s=1, c='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D spikes')
    ax.set_xlim(xmin=0, xmax=resolution[0])
    ax.set_ylim(ymin=0, ymax=resolution[1])
    
    ani = animation.FuncAnimation(fig, updatefig, interval=timestep//50, blit=True)
    plt.show()