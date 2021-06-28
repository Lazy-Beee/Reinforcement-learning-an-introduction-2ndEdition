import numpy as np

x = 0
for i in range(100000):
    x += np.random.normal(0, 0.01)

print(x/100000)