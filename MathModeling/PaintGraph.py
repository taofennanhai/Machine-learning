import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


img = Image.open('Q2Data/10cm.png')


image1 = Image.open('Q2Data/10cm.png')

image2 = Image.open('Q2Data/40cm.png')

image3 = Image.open('Q2Data/100cm.png')

image4 = Image.open('Q2Data/200cm.png')



plt.subplot(221)

plt.imshow(image1)

plt.subplot(222)

plt.imshow(image2)


plt.subplot(223)

plt.imshow(image3)

plt.subplot(224)

plt.imshow(image4)

plt.xticks(alpha=0)
plt.tick_params(axis='x', width=0)
plt.xticks(alpha=0)
plt.tick_params(axis='y', width=0)

plt.show()