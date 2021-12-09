import numpy as np
import torch


index = torch.from_numpy(np.array([[1, 2, 0], [2, 0, 1]])).type(torch.LongTensor)
index = index[:, :, None].expand(-1, -1, 10)
print(index)
