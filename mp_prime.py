import numpy as np
from multiprocessing import Pool
from time import time

if __name__ == '__main__':

    start = int(5e6) # Smallest prime candidate to check
    N = 100 # Number of candidates to check

    # Brute force prime checking
    def check_prime(n, disp=True):
        prime = True
        for j in range(2, n):
            if n%j == 0:
                prime = False
        if disp: print(n, prime)
        return prime
    
    # Single process
    print('=== Single Process ===')
    t0 = time()
    for i in range(start, start+N):
        prime = check_prime(i)
    tf = time()
    tS = tf-t0

    # Multiprocess
    print('=== Multi Process ===')
    t0 = time()
    with Pool(processes=12) as p:
        p.map(check_prime, range(start, start+N))
    tf = time()
    tM = tf-t0

    # Print results
    print('')
    print('Single process: {}s'.format(tS))
    print('Multi process: {}s'.format(tM))
    print('Single/Multi: {}'.format(tS/tM))
