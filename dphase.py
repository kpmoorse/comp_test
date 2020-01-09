import numpy as np
import matplotlib.pyplot as plt

# Estimate phase difference between two signals using Taylor expansion
# Works well for low-degree polynomials; not for quickly-varying functions

dt = 0.01
t = np.arange(0,10,dt)

f = (t-5)**2
g = (t-6)**2

dgdt = np.diff(g)/dt
dgdt = np.insert(dgdt,0,dgdt[0])
d2gdt2 = np.diff(np.diff(g))/dt**2
d2gdt2 = np.insert(d2gdt2,[0,-1],[d2gdt2[-0],d2gdt2[-1]])

G = np.hstack((np.ones((t.shape[0],1)),
                g[:,None],
                dgdt[:,None],
                d2gdt2[:,None]))

b = np.linalg.lstsq(G, f, rcond=None)[0]
print(b)

plt.plot(t,f,t,g)
# plt.plot(t,f,t,g,t,dgdt,t,d2gdt2)
plt.show()