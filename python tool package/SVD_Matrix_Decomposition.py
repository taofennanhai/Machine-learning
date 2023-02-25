import numpy as np

matirx = ([[1, 2, 3], [4, 5, 6]])

U, sigma, V = np.linalg.svd(matirx, full_matrices=False)

print("U sigma V矩阵")
print(U)
print(sigma)
print(V)

print("U sigma V矩阵的形状")
print(U.shape)
print(sigma.shape)
print(V.shape)

print("矩阵还原")
print(np.dot(np.dot(U, np.diag(sigma)), V))

print('yes',0.1)