from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('/home/badelekpi/Documents/priv_scripts/Emotion-Recognition-master-backup/testy_konwersji/WIN_20230326_21_34_15_Pro.jpg')
avg_img= imread('/home/badelekpi/Downloads/pseudo_average.jpg')
mean_pool=block_reduce(img, block_size=(9,9,1), func=np.mean)
max_pool=block_reduce(img, block_size=(9,9,1), func=np.max)
min_pool=block_reduce(img, block_size=(9,9,1), func=np.min)

plt.figure(1)
plt.subplot(221)
imgplot = plt.imshow(img)
plt.title('Obraz wejściowy')

plt.subplot(222)
imgplot3 = plt.imshow(avg_img)
plt.title('Funkcja Average pooling')

plt.subplot(223)
imgplot1 = plt.imshow(max_pool)
plt.title('Funkcja Max pooling')

plt.subplot(224)
imgplot1 = plt.imshow(min_pool)
plt.title('Funkcja Min pooling')

plt.show()