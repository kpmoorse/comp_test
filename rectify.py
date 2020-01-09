import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import cv2


gdim = (7,5)
s = 0.1
ctr = (0,0,0.5)
rot = (0,0)
r = R.from_euler('xy', (45, 30), degrees=True)
cams = [(-0.25,0,0), (0.25,0,0)]


# Generate a grid of points in 3D space
def gen_grid(gdim, s):

    n,m = gdim

    grid = np.array(list(itertools.product(range(n),range(m)))) * s
    xy_off = np.tile([(n-1)*s/2, (m-1)*s/2, 0], (n*m,1))

    grid = np.hstack((grid,np.zeros((n*m,1))))
    grid -= xy_off

    return grid


# Apply translation and rotation transforms
def tf_grid(grid, ctr, rot):

    nm = grid.shape[0]
    xyz_off = np.tile(ctr, (nm,1))

    grid = rot.apply(grid)
    grid += xyz_off

    return grid

# Project 3D points onto 2D image plane
def project(cam, pts):

    pts = pts - np.tile(cam, (pts.shape[0],1))
    x0, y0, z0 = pts.T
    x = x0/z0
    y = y0/z0
    return np.hstack((x[:,None], y[:,None]))

def append_ones(mat):
    return np.hstack((mat,np.ones((mat.shape[0],1))))

grid = gen_grid(gdim, s)
grid = tf_grid(grid, ctr, r)
nplots = len(cams)+1
x, y, z = grid.T

fig = plt.figure()
ax = fig.add_subplot(1,nplots,1, projection='3d')
ax.plot(x,y,z,'.')
for cam in cams:
    ax.scatter(*cam)
    pass

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([0,2])

proj = []
for i, cam in enumerate(cams):
    proj.append(project(cam, grid))
    ax = fig.add_subplot(1,nplots,i+2)
    xx, yy = proj[-1].T
    ax.plot(xx, yy, '.')
    if i==1:
        H,_ = cv2.findHomography(*proj)
        F,_ = cv2.findFundamentalMat(*proj)
        _,H1,H2 = cv2.stereoRectifyUncalibrated(*proj, F, (1280,800))
        print(H)
        ax.plot(*append_ones(proj[0]).dot(H)[:,:2].T, '.')




plt.show()
