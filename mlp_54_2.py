import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-8, 8, 100)
y1 = 1/(1+np.exp(-x))
y2 = y1 * (1-y1)
plt.plot(x, y1, "r-")
plt.plot(x, y2, "g-")
plt.show()

np.array
