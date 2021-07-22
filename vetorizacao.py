import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tempo1 = time.time()
c = np.dot(a, b)
tempo2 = time.time()

print(c)
print('Com Vetorização: ' + str(1000*(tempo2 - tempo1)) + ' milisegundos')  # ~ 1.5 ms

c = 0
tempo1 = time.time()
for i in range(1000000):
    c += a[i] * b[i]
tempo2 = time.time()

print(c)
print('Com Loop For: ' + str(1000*(tempo2 - tempo1)) + ' milisegundos')  # ~ 450 ms
