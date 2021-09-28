import numpy as np
import Kmeans聚类方法 as kmeans

def biKmeans(dataSet, k):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))    # 每个点对质心的聚类的集合

    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    print(np.mat(centroid0))

    for j in range(m):    # 计算每个点到质心的误差距离
        clusterAssment[j, 1] = kmeans.VecDistance(np.mat(centroid0), dataSet[j, :])**2

    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]    # 获取当前簇的所有数据

            centroidMat, splitClustAss = kmeans.Kmeans(ptsInCurrCluster, 2)    # 对该簇的数据进行K-Means聚类

            sseSplit = sum(splitClustAss[:, 1])    # 该簇聚类后的sse

            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])    # 获取剩余收据集的sse

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
            #    将簇编号0,1更新为划分簇和新加入簇的编号

        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)    # 把新划分的簇重新编号
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClustAss is: ", len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :]    # 增加质心
        centList.append(bestNewCents[1, :])

        # 更新簇的分配结果
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment




if __name__ == '__main__':
    dataSet = kmeans.loadDataSet('ex00.txt')
    biKmeans(np.array(dataSet), 3)