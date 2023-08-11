import numpy as np
from matplotlib import pyplot as plt

img_array = np.load("X_test.npy")
plt.imshow(img_array[597], cmap="gray")
plt.show()