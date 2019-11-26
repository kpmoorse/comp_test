import numpy as np
import matplotlib.pyplot as plt

# Linearized pendulum
A = np.array([[0,1],[-1,0]])

th = np.arange(-np.pi/4, np.pi/4, 0.1)
dthdt = th.copy()
TH, dTHdT = np.meshgrid(th, dthdt)

d2THdT2 = -TH

plt.quiver(TH, dTHdT, dTHdT, d2THdT2)
plt.set_xlabel('$\\theta$')
plt.xlabel('$\\theta$')
plt.show()