import matplotlib.pyplot as plt
import numpy as np

import h5py
import parameters
# Parameters View
#parameters.view()

x1 = np.random.randn(50)
x2 = np.random.randn(50)

plt.plot(x1, 'b')
plt.plot(x2, 'r')
plt.ylabel('Accuracy')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =")
plt.legend(("test", "test2"))
plt.show()



