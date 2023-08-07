import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, +2, 0.1)

i_x = x<0
y = np.copy(x)
y[i_x] = 0

fig, ax = plt.subplots()
ax.plot(x, y,'o-')
plt.xlabel('x'); plt.ylabel('y')
plt.title('ReLU')
plt.grid(axis='both', color='0.95')
plt.show()
