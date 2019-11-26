import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

max_iters = 500
dim = (100,100)
r = 0.5
plt.figure()

bc = 'constant' # Dirichlet boundary (f(x)=0)
# bc = 'edge' # Neuman boundary (f'(x)=0)

u_last = np.zeros(dim)
u_now = np.zeros(dim)

u_now = np.pad(u_now, 1, bc)

if len(dim) == 1:

    for i in range(max_iters):

        u_next = r**2 * u_now[2:] + \
            2*(1-r**2) * u_now[1:-1] + \
            r**2 * u_now[0:-2] + \
            -u_last

        u_last = u_now[1:-1]
        u_now = np.pad(u_next, 1, bc)

        u_now[50] = np.sin(2*np.pi*i/100)

        plt.clf()
        plt.plot(u_now)
        plt.ylim([-1,1])
        plt.draw()
        plt.pause(0.01)

elif len(dim) == 2:

    kernel = np.array([[0,r**2,0],[r**2,0,r**2],[0,r**2,0]])
    X, Y = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    mask = (Y==50) & ((X<49) | (X>51)) # Single slit

    for i in range(max_iters):

        # u_next = (2-4*r**2) * u_now[1:-1, 1:-1] + \
        #     r**2 * u_now[2:, 1:-1] + \
        #     r**2 * u_now[0:-2, 1:-1] + \
        #     r**2 * u_now[1:-1, 2:] + \
        #     r**2 * u_now[1:-1, 0:-2] + \
        #     -u_last

        u_next = (2-4*r**2) * u_now[1:-1, 1:-1] + \
            ss.convolve2d(u_now, kernel, mode='same')[1:-1, 1:-1] + \
            -u_last

        u_last = u_now[1:-1, 1:-1]

        u_next[mask] = 0
        u_next[75,50] = np.sin(2*np.pi*i/10) # Oscillating source

        u_now = np.pad(u_next, 1, bc)

        show = u_next
        show[mask]=-1

        plt.clf()
        plt.imshow(show, vmin=-1, vmax=1)
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)