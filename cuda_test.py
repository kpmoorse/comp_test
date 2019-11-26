import numpy as np
import matplotlib.pyplot as plt
import torch
from time import time
from tqdm import tqdm

N = 1000 # Number of iterations
n = 1000 # Matrix dimension

tN = [] # Numpy results
tT = [] # (Py)Torch results

for i in tqdm(range(N)):

    A = np.random.random((n,n))
    B = np.random.random((n,n))

    t0 = time()
    tempN = np.dot(A,B)
    tf = time()

    tN.append(tf-t0)

    A, B = (torch.from_numpy(x).cuda() for x in (A,B))

    t0 = time()
    tempT = torch.mm(A,B)
    tf = time()

    tT.append(tf-t0)

print("Numpy (mean): {}s".format(np.mean(tN)))
print("PyTorch (mean): {}s".format(np.mean(tT)))

bins = np.linspace(0, np.max(tN+tT), num=100)

plt.hist((np.log(tN), np.log(tT)))
plt.legend(['Numpy', 'PyTorch'])
plt.xlabel('Run Time (log[s])')
plt.show()