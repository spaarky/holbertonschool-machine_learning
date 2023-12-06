#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# y for the slope of the curve and c='r' for the red color of the curve
plt.plot(y, c='r')

plt.autoscale(axis='x', tight=True)

# display figure
plt.show()
