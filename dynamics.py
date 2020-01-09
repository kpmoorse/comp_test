import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import sys

x0 = np.array([[pi/8], [0]])
g = 10
b = 3
dt = 0.01
A = np.array([[0,1],[-g,-b]])

fig = plt.figure()
fig.canvas.mpl_connect('close_event', sys.exit)
x = x0
for _ in range(500):

    xdot = A.dot(x0)
    x += xdot*dt

    plt.clf()
    plt.plot([0,sin(x0[0])], [0,-cos(x0[0])], 'o-')
    plt.gca().axis('equal')
    plt.gca().set(xlim=(-0.6,0.6), ylim=(-1.2, 0))
    plt.draw()
    plt.pause(0.01)