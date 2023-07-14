import numpy as np
import gzip


# Another way to generate distribution
distribution = []
iters = 1_000_000
max_targets = 1200
n = 10_000

for target in range(1, max_targets + 1):
    arr = []
    for i in range(iters):
        amr = (np.mean(np.random.choice(n, target, replace=False)) - 0.5) / n
        imr = 1 - amr
        arr.append(np.min([amr, imr]))
    distribution.append(arr)

distribution = np.array(distribution)

f = gzip.GzipFile('../../data/distribution_compressed.npy.gz', "w")
np.save(file=f, arr=distribution)
f.close()