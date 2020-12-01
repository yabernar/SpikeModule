from brian2 import *
from matplotlib import pyplot
import numpy as np


class ShapeNetwork:
    def __init__(self):
        # Simulation parameters
        self.resolution = (640, 480)
        tau = 10 * ms
        eqs = '''
        dv/dt = (1-v)/tau : 1
        '''

        # Create input Layer (DO NOT USE SELF WITH BRIAN OBJECTS)
        G = NeuronGroup(np.prod(self.resolution), eqs, threshold='v > 0.8', reset='v=0', method='linear')
        G.v = 'rand()'
        G.v -= 0.20
        S = SpikeMonitor(G)
        M = StateMonitor(G, 'v', record=True)

        # Create processing Layer

        # Connecting two layers with synapses

        # Run simulations
        print('Before v = %s' % G.v[0])
        run(1 * us)
        print('After v = %s' % G.v[0])
        # plot the trace of neuron 3:
        # plot(mon.t / ms, mon.v[3])
        # xlabel('Time (ms)')
        # ylabel('Neuron index')
        # title('N = 100 Neuron Population spikes')
        # show()

        self.return_spike_accumulation(S)

    def run_input_sequence(self, sequence):
        pass

    def return_spike_accumulation(self, Spikes):
        x, y = Spikes.i[:]%self.resolution[0], Spikes.i[:]//self.resolution[0]

        plot(x, y, '.k')
        xlabel('X')
        ylabel('Y')
        title('2D spikes')
        show()

    def circle_test(self):
        half_diameter = 0.2



if __name__ == '__main__':
    sn = ShapeNetwork()
