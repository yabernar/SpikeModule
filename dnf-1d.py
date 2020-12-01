import numpy as np
from scipy import signal, special
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gaussian(distance, sigma):
    return np.exp(-(distance / sigma) ** 2 / 2)


def gaussian_distribution(position, size, sigma):
    result = np.empty(size)
    for i in range(size):
        current = (i / size)
        result[i] = gaussian(np.abs(position - current), sigma)
    return result


def gaussian_distribution_uniform(position, size, sigma):
    result = gaussian_distribution(position, size, sigma)
    result = np.divide(result, np.sum(result))
    return result


class DNF1D:
    def __init__(self, width):
        self.width = width
        self.dt = 0.1
        self.tau = 0.5
        self.dt_tau = self.dt/self.tau
        self.cexc = 2
        self.sexc = 0.01
        self.cinh = 1
        self.sinh = 0.1
        self.gain = 2
        self.resting_level = -2
        self.gi = 0

        self.input = np.zeros(width, dtype=float)
        self.potentials = np.zeros(width, dtype=float)
        self.activations = np.zeros(width, dtype=float)
        self.lateral = np.zeros(width, dtype=float)
        self.kernel = np.zeros(width*2-1, dtype=float)

        self.kernel = (self.cexc * gaussian_distribution_uniform(0.5, self.width*2-1, self.sexc)) - (self.gi / (self.width*2-1))
        # (self.cinh * gaussian_distribution_uniform((0.5, 0.5), (self.width*2, self.height*2), self.sinh)) - \

    def normal_distance(self, a, b):
        return np.abs((a - b) / self.width)

    def difference_of_gaussian(self, x, y):
        return self.kernel[np.abs(x - y)]

    def gaussian_activity(self, a, b, sigma):
        for i in range(self.width):
            self.input[i] = gaussian(self.normal_distance(a*self.width, i), sigma) + gaussian(self.normal_distance(b*self.width, i), sigma)

    def update_neuron(self, x):
        # lateral = 0
        # for i in range(self.width):
        #     for j in range(self.height):
        #         lateral += self.potentials[i, j]*self.optimized_DoG((i, j), x)
        self.potentials[x] += self.dt_tau * (-self.potentials[x] + self.resting_level + self.lateral[x] + self.input[x]*self.gain)
        # if self.potentials[x] > 1:
        #     self.potentials[x] = 1
        # elif self.potentials[x] < -1:
        #     self.potentials[x] = -1

    def update_map(self):
        self.activations = special.expit(self.potentials)
        self.lateral = signal.fftconvolve(self.activations, self.kernel, mode='same')
        # self.lateral = np.divide(self.lateral, self.width*self.height)*40*40

        # print(self.lateral)
        neurons_list = list(range(self.width))
        np.random.shuffle(neurons_list)
        for i in neurons_list:
            self.update_neuron(i)


def updatefig(*args):
    dnf.update_map()
    pot.set_ydata(dnf.potentials)
    return pot,


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_ylim(-5, 5)
    dnf = DNF1D(45)
    dnf.gaussian_activity((0.2), (0.65), 0.1)
    # krn, = ax.plot(dnf.kernel)
    inp, = ax.plot(dnf.input)
    pot, = ax.plot(list(range(dnf.width)), dnf.input)
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)

    plt.show()