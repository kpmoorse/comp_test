import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors as NN


class Swarm(object):

    def __init__(self, N=100):

        self.dt = 0.025
        self.x = np.random.random((N,2))*2-1
        th = np.random.random((N,1))*2*pi-pi
        self.v = np.hstack((cos(th),sin(th)))

    def step(self):

        # Apply Gaussian noise to heading angle and step position
        mv = np.tile(np.linalg.norm(self.v,axis=1)[:,None], (1,2))
        th = np.arctan2(self.v[:,1],self.v[:,0])[:,None]
        th += np.random.normal(0,0.25,th.shape)
        self.v = np.hstack((cos(th),sin(th)))*mv
        self.x += self.v*self.dt

        # Agents reflect on edge intercept
        self.v[np.abs(self.x[:,0])>1,0] *= -1
        self.v[np.abs(self.x[:,1])>1,1] *= -1

    def disp(self):

        x,y = self.x.T
        plt.clf()
        plt.plot(x,y,'.')
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        plt.draw()
        plt.pause(0.025)


class Noise(object):

    def __init__(self):

        self.x = []

    def step(self, x):

        # Shuffle order
        self.x = x.copy()
        np.random.shuffle(self.x)

        # Append erroneous points
        self.x = np.vstack((self.x, np.random.random((5,2))*2-1))


class Tracker(object):

    def __init__(self):

        self.x = [] # [3xNx2] Measured object positions
        self.v = [] # [Nx2] First-order estimate of object velocities
        self.c = [] # [2xNx2] Composition maps between object sets: x[i][c[i]]=x[i+1]

    def step(self, pos):

        self.x.append(pos.copy())
        
        # *** This section needs to be cleaned up ***
        if len(self.x)>3:
            self.x.pop(0)
            self.c.append(self.knn(self.x[1], self.x[2]))
            self.c.pop(0)
            self.v = self.x[1] - self.x[0][self.c[0],:]
        elif len(self.x)>=2:
            self.c.append(self.knn(self.x[-2], self.x[-1]))

    def knn(self, vin, vout, thresh=None):

        nn = NN()
        nn.fit(vin)
        res = nn.kneighbors(vout, n_neighbors=1)
        if thresh:
            ix = res[1].T[0][res[0].T[0]<thresh]
        else:
            ix = res[1].T[0]
        return ix

    def disp(self):

        if len(self.v)==0:
            print("Velocity data is unavailable; ignoring display command.")
            return

        x_old,y_old = self.x[0][self.c[0],:][self.c[1],:].T
        x_new,y_new = self.x[1].T
        vx,vy = self.v[self.c[1],:].T
        # vx,vy = self.v.T
        plt.clf()
        plt.plot(x_new,y_new,'.')
        plt.plot(x_old+vx,y_old+vy,'o',fillstyle='none')
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        plt.draw()
        plt.pause(0.1)
    

if __name__ == '__main__':

    swarm = Swarm(N=10)
    noise = Noise()
    tracker = Tracker()
    for i in range(100):
        noise.step(swarm.x)
        tracker.step(noise.x)
        if len(tracker.v)>0:
            tracker.disp()
        # swarm.disp()
        swarm.step()
