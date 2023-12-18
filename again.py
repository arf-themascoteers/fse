import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
ind = np.array([0, 3, 4])

arr[arr not in ind] = 0


print(arr)