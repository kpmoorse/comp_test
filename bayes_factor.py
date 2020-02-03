import numpy as np
from numpy import sqrt, exp, pi, log, sum
import matplotlib.pyplot as plt
from tqdm import tqdm

class BayesTest(object):

    def __init__(self, dist=0, n_data=50, n_bs=1e4, plot=True, progress=True):

        self.dist = dist

        # Define Gaussian and color functions
        N = lambda x, x0, s: (s*sqrt(2*pi))**-1 * exp(-(x-x0)**2/(2*s**2))
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Compute cartesian coordinates
        x = np.arange(-1, 1, 0.01)
        y = np.arange(-1, 1, 0.01)
        X, Y = np.meshgrid(x, y)

        # Define PDFs
        M1 = lambda x, y: N(x, -0.25, 0.25) * N(y, 0, 0.5)
        M2 = lambda x, y: N(x, 0, 0.5) * N(y, -0.25, 0.25)

        # Generate discrete PDF samples for plotting
        M1d = M1(X,Y)
        M2d = M2(X,Y)

        # Generate data
        n = n_data
        if self.dist == 0:
            # Draw from neutral distribution
            data = np.hstack((
                np.random.normal(0, 0.25, n)[:,None],
                np.random.normal(0, 0.25, n)[:,None]
            ))
        elif self.dist == 1:
            # Draw from M1
            data = np.hstack((
                np.random.normal(-0.25, 0.25, n)[:,None],
                np.random.normal(0, 0.5, n)[:,None]
            ))
        elif self.dist == 2:
            # Draw from M2
            data = np.hstack((
                np.random.normal(0, 0.5, n)[:,None],
                np.random.normal(-0.25, 0.25, n)[:,None]
            ))
        
        # Compute Bayes factor
        B12 = sum(log(M1(data[:,0], data[:,1]) / M2(data[:,0], data[:,1])))/sqrt(n_data)
        # print('Bayes Factor (B12): '+str(B12))

        # Compute bootstrapped Bayes factor (bsB) distribution
        bsB = []n_b
            logB = sum(log(M1(subset[:,0], subset[:,1]) / M2(subset[:,0], subset[:,1])))/sqrt(n_data)
            bsB.append(logB)
        bsB = np.array(bsB)
        self.mean_bsB = np.mean(bsB)
        # self.mean_bsB = B12

        if plot:
            # Plot results
            blues = self.color_fade(self.hex2rgb(c[0]), 10)
            oranges = self.color_fade(self.hex2rgb(c[1]), 10)

            fig, ax = plt.subplots()

            ax.contour(X, Y, M1d, 10, colors=blues)
            ax.contour(X, Y, M2d, 10, colors=oranges)

            xx, yy = data.T
            ax.plot(xx, yy, 'k.')

            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_aspect('equal', 'box')

            plt.figure()
            plt.hist(bsB)

            plt.show()

    # Get RGB values from hexadecimal color
    def hex2rgb(self, hexval):
        r = int(hexval[1:3], 16)/255.
        g = int(hexval[3:5], 16)/255.
        b = int(hexval[5:], 16)/255.
        return [r,g,b]

    # Generate list of fading colors
    def color_fade(self, c, n):
        c_list = [c]
        for i in range(n-1):
            c_list.append([1-(1-x)*3/4 for x in c_list[-1]])
        return c_list[::-1]


if __name__ == '__main__':

    sig = lambda x: (1+exp(-x))**-1

    n_list = np.arange(1, 1000, 10)
    mean_list = np.zeros(n_list.shape)
    for i,n in tqdm(enumerate(n_list)):
        bt = BayesTest(dist=0, n_data=n, plot=False, progress=False)
        mean_list[i] = bt.mean_bsB
    plt.subplot(211)
    plt.plot(n_list, mean_list, '.')
    plt.subplot(212)
    plt.plot(n_list, sig(mean_list), '.')
    plt.show()