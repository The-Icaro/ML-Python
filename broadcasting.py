import numpy as np

a = np.random.randn(12288, 150)
b = np.random.randn(120, 45)
c = np.dot(a,b)
print(c.shape)