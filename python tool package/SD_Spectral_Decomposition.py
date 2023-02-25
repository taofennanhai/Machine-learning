import numpy as np

matirx = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

print('要分解的方阵:')
print(matirx)

e_vals, e_vecs = np.linalg.eig(matirx)

print('特征值:')
print(e_vals)
print('特征向量矩阵:')
print(e_vecs)

print("矩阵还原")
print(np.dot(e_vecs, np.dot(np.diag(e_vals), np.linalg.inv(e_vecs))))
