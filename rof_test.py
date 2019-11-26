import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle as rof
from tqdm import tqdm
from math import pi, sqrt, exp

# Repeated smoothing window for derivatives
def repsmooth(x,w,n):

    assert ((w-1)/2)%1 == 0, "Smothing width must be an odd integer"

    hw = int((w-1)/2)
    for i in range(n):
        x = np.convolve(np.pad(x,hw,mode='edge'), np.ones(w)/w, 'valid')
    return x

def gauss(n=11,sigma=1):

    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

def repkernel(x,k,n):

    for i in range(n):
        x = np.convolve(x,k,'same')
    return x

dt = .001
t = np.arange(0,1,dt)

# Calculate smooth derivative
x = np.sin(2*np.pi*5*t) * (1-np.sin(2*np.pi*0.5*t))
x += 5*(t-0.5)**2
dx = np.diff(x)/dt

# Calculate noisy derivative
xn = x + np.random.normal(0,0.5,x.shape)
dxn = np.diff(xn)/dt

# Apply denoising methods
dy_rof = rof(dxn,7000)
dy_rsm = repsmooth(dxn,43,2)
dy_rpk = repkernel(dxn, gauss(n=86,sigma=13), 2)

# Plot results
plt.plot(t[:-1],dxn,t[:-1],dy_rof,t[:-1],dy_rsm,t[:-1],dy_rpk,t[:-1],dx,'--')
plt.legend(['Noisy','ROF','Smooth','Gaussian','Original'])
plt.show()

# Grid search for best ROF parameter
def search_rof():
    result = []
    for l in tqdm(np.arange(1000,10000)):
        dy_rof=rof(dxn,l)
        res = np.sum((dy_rof-dx)**2)
        result.append([l,res])
    result = np.array(result)
    plt.plot(result[:,0], result[:,1],'.-')
    plt.show()

#Grid search for best smoothing parameter
def search_rsm():
    result = []
    for w in tqdm(np.arange(1,101,2)):
        hw = int((w-1)/2)
        dy_rsm=repsmooth(dxn,w,2)
        res = np.sum((dy_rsm[hw:-hw]-dx[hw:-hw])**2)
        result.append([w,res])
    result = np.array(result)
    plt.plot(result[:,0], result[:,1],'.-')
    plt.show()

#Grid search for best kernel parameter
def search_rpk():
    result = []
    for w in tqdm(np.arange(1,100)):
        dy_rpk=repkernel(dxn,gauss(n=w,sigma=13),2)
        res = np.sum((dy_rpk-dx)**2)
        result.append([w,res])
    result = np.array(result)
    plt.plot(result[:,0], result[:,1],'.-')
    plt.show()

# search_rof()
# search_rsm()
# search_rpk()