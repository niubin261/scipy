import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
import numpy as np
iris = datasets.load_iris();
dataMat=iris.data

covMat=np.cov(dataMat.T)
corMat=np.corrcoef(dataMat.T)

avg=np.mean(dataMat, axis=0)
m, n=np.shape(dataMat)
meanRemoved=dataMat - np.tile(avg, (m,1))
normData=meanRemoved/(np.std(dataMat))
cov=np.cov(normData.T)
eigValue=np.linalg.eig(cov)[0]
eigVector= np.linalg.eig(cov)[1]
eigValId=np.argsort(-eigValue)
selectVec=np.matrix(eigVector.T[:4])
finalData=normData*selectVec.T
print finalData



fig = plt.figure(figsize=(8, 8))
bplot = plt.boxplot(dataMat,
                    notch=False,
                    sym='rs',
                    vert=True)

plt.xticks([1,2,3,4],['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
t = plt.title('Iris data box plot')
plt.show()


