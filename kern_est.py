import numpy as np
import matplotlib.pyplot as plt
from fourier import fft, ifft
from rof_test import gauss
from tqdm import tqdm

# Estimate the kernel relating two signals via the convolution theorem

def kernel(F,G,t=None):

    assert len(F) == len(G), "Input arrays must have equal length"
    if len(t) == 0: t = np.arange(len(F))

    # Calculate the fourier transforms of F and G
    FF, f = fft(F,t)
    FG, _ = fft(G,t)

    # Calculate the inverse fourier transform of the quotient
    k, tt = ifft(FG/FF, f)
    k = np.real(k)

    # Sort results by time value
    sort = np.argsort(tt)
    k = k[sort]
    tt = tt[sort]

    return k, tt

if __name__ == '__main__':

    dt = 0.001
    t = np.arange(0,1200,dt)

    # gauss = lambda x,x0,s: np.exp(-(x-x0)**2/(2*s**2))

    # events = [0.1,0.3,0.6,0.85]
    s = 0.01
    e = np.random.random(10000)*np.max(t) #events
    ix = np.floor((e-t[0])/dt).astype(int)
    F = np.zeros(t.shape)
    F[ix] = 1

    re = e + np.random.normal(0, s, e.shape)
    rix = np.floor((re-t[0])/dt).astype(int)
    G = np.zeros(t.shape)
    G[rix] = 1

    # F = np.zeros(t.shape)
    # for e in tqdm(events):

    #     F[np.argmin(np.abs(t-e))] = 1

    # G = np.zeros(t.shape)
    # for e in tqdm(events):
    #     re = e + np.random.normal(0,s)
    #     G[np.argmin(np.abs(t-re-0.05))] = 1


    k, tt = kernel(F,G,t)
    w = 11
    ksm = np.convolve(k, gauss(n=35, sigma=5), 'same')
    # k, tt = k[tt>=0], tt[tt>=0]

    plt.subplot(211)
    plt.plot(t,F,t,G)
    plt.subplot(212)
    plt.plot(tt, k, tt, ksm)
    plt.show()