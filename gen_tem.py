import numpy as np
from numpy import sqrt, exp
import matplotlib.pyplot as plt
import scipy.special as ss

H = lambda x, n: ss.hermite(n).__call__(x)

def hg(x, y, n=0, m=0, w=1):

    E = H(sqrt(2)*x/w,n) * exp(-(x/w)**2) * H(sqrt(2)*y/w,m) * exp(-(y/w)**2)
    return E**2

x = np.arange(-2,2,0.01)
y = x.copy()
X, Y = np.meshgrid(x,y)

TEM = hg(X,Y,n=1,m=1)
cTEM = np.fft.fftshift(TEM)
# FTEM = np.abs(np.fft.fft2(cTEM))
# print(FTEM)
# print(FTEM.dtype)
# print(FTEM.shape)

plt.imshow(TEM)
plt.show()
