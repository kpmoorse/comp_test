import numpy as np
from numpy import sqrt, pi, exp
import cv2
import matplotlib.pyplot as plt
import scipy.signal as ss
from multiprocessing import Process, Queue

# Generate a 2D Gaussian kernel
def gk2d(sigma, n):

    # gauss = lambda x,x0,s: (s*sqrt(2*pi))**-1 * exp(-(x-x0)**2/(2*s**2))
    gauss = lambda x,x0,s: exp(-(x-x0)**2/(2*s**2))
    x = np.arange(n)
    y = gauss(x,(n-1)/2,sigma)
    kernel = y[:,None].dot(y[None,:])
    return kernel / np.sum(kernel)

def lpf(new, old, alpha):
    return alpha*new + (1-alpha)*old

def process(fname):

    # Get video and initialize VideoCapture object
    cap = cv2.VideoCapture(fname)
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    fr = 60 # Frame rate
    ft = int(1000/fr) # Frame time
    kernel = gk2d(3,7)

    # Initialize background subtraction object
    backSub = cv2.createBackgroundSubtractorMOG2(history=int(1e6), varThreshold=5)
    backSub2 = cv2.createBackgroundSubtractorKNN()

    fgm2_hold = []

    t0 = 0 # Seconds
    i = 0

    # Loop over video frames
    while cap.isOpened():

        ret, frame = cap.read()
        flag = True
        if ret == True:

            if i<t0*fr:
                i += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[200:600,:640]

            # On first frame, generate empty image
            if flag:
                empty = np.zeros(frame.shape).astype('uint8')
                fgm2 = empty
   
            bgs = backSub.apply(frame)
            # bgs2 = backSub2.apply(frame)

            fgm1 = cv2.morphologyEx(bgs, cv2.MORPH_OPEN, np.ones([3]*2))
            fgm2 = lpf(ss.convolve2d(bgs, kernel, 'same').astype('uint8'), fgm2, 0.5).astype('uint8')
            _, fgm2 = cv2.threshold(fgm2, 20, 255, cv2.THRESH_BINARY)

            # for x in [frame, fgm1, fgm2, empty]:
            #     print(x.dtype)

            disp = np.block([
                [frame, fgm1],
                [fgm2, empty]
            ])
            # disp = cv2.pyrDown(disp)

            # Display processed frame
            cv2.imshow('Video', disp)
            if cv2.waitKey(ft) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    process("/home/kellan/git/comp_test/2019-12-06_cam00.mp4")
    # process("/mnt/dlab/git/cam_array/video/2019-12-09_cam00.mp4")
    # print(gk2d(1,5))