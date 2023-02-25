import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

matrix = np.array([[1, 0.5, 0+0.5j],
                   [0.5, 3, 0],
                   [0, 1, 5]], dtype=complex)


fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(matrix.shape[0]):

    r = matrix[i][i]
    radius = 0
    for j in range(matrix.shape[0]):
        if i != j:
            radius = radius + abs(matrix[i][j])    # 求模长
    ax.add_patch(Circle(xy=(abs(matrix[i][i]), 0), radius=radius, alpha=0.2+0.1*i))

plt.axis('scaled')
plt.axis('equal')

plt.show()
print("有三个不同的实特征根")