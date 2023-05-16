import numpy as np
import math, sys

filename = sys.argv[1]
metrix = np.loadtxt(filename)


def cal_entropy(m):
    sum_err = m.sum()
    entropy = 0.0
    if sum_err == 0:
        return entropy
    for i in np.nditer(m):
        if i != 0:
            p = i / sum_err
            entropy += -1 * (p * math.log2(p))
    return entropy


print(cal_entropy(metrix))
