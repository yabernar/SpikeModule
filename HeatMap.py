from brian2 import *
from matplotlib import pyplot
import numpy as np
from dv import AedatFile
from PIL import Image

np.set_printoptions(threshold=np.inf)

with AedatFile("Captures/circle_travel.aedat4") as f:
    # Access dimensions of the event stream
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])
timescale = 50
offset = 2000000
start_time = events[0][0] + offset
current_index = 0
while current_index < events.shape[0] - 1 and events[current_index][0] < start_time:
    current_index += 1

# Simulation parameters
resolution = (640, 480)
tau = 10*ms
El = -0.7

# LIF
eqs_LIF = '''dv/dt = ((El - v) + I)/tau : 1
        I : 1'''
# Not LIF
eqs_2 = '''dv/dt = (1-v)/tau : 1'''

# Create input Layer (DO NOT USE SELF WITH BRIAN OBJECTS)
G = NeuronGroup(np.prod(resolution), eqs_2, threshold='v > 0.8', reset='v = 0', method='linear')
G.v = 0
# G.I = 0  # 'rand()'
# G.v /= 0.8
# network = Network(G, network_operation)
S = SpikeMonitor(G)
M = StateMonitor(G, 'v', record=True)

# Create processing Layer

# Connecting two layers with synapses

# plot the trace of neuron 3:
# plot(mon.t / ms, mon.v[3])
# xlabel('Time (ms)')
# ylabel('Neuron index')
# title('N = 100 Neuron Population spikes')
# show()


@network_operation(dt=timescale * us)
def update_stimulus(t):
    global current_index, G
    tick = int(t.variable.get_value()*1000000)  # time as us
    print(tick)
    current_end = start_time + tick
    while current_index < events.shape[0] - 1 and events[current_index][0] < current_end:
        G[(resolution[1] - events[current_index][2] - 1) * resolution[0] + events[current_index][1]].v += 0.4
        current_index += 1


def return_spike_accumulation(spikes):
    x, y = spikes.i[:] % resolution[0], spikes.i[:] // resolution[0]
    plot(x, y, '.k')
    plt.xlim(xmin=0, xmax=resolution[0])
    plt.ylim(ymin=0, ymax=resolution[1])
    xlabel('X')
    ylabel('Y')
    title('2D spikes')
    show()


if __name__ == '__main__':
    # Run simulations
    run(1000 * us)
    return_spike_accumulation(S)