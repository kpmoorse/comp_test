import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

iters = 100000
n_max = 50

result = []
for n in tqdm(np.arange(n_max)+1):
    subres = []
    for i in range(iters):
        players = np.arange(n)+1
        draws = players.copy(); np.random.shuffle(draws)
        sd = np.any(players==draws)
        subres.append(sd)
    result.append(np.mean(subres))
print(result)
plt.plot(np.arange(n_max)+1, result)
plt.show()