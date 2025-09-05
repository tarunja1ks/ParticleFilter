import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

overall_mean = np.mean(arr, axis=0)
print(f"Overall mean: {overall_mean}")