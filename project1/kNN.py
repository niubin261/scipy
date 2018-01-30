from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import cross_validation
wines = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None)

wines = np.array(wines)

labelsArr = wines[:,0:1]

labelsTmp = labelsArr.tolist()
labels = []
length = len(labelsTmp)
for i in range(len(labelsTmp)):
    labels.append(labelsTmp[i][0])


dataSet=wines[:,1:]
'''
test for preprocessing
'''
'''
print len(wines)
print wines.shape
print labelsArr.shape
print labelsArr

print length
print labelsTmp[177][0]
print labelsTmp
print labels
print dataSet.shape
'''

def KNN(inY,Ylabel,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inY,(dataSetSize,1)) -dataSet
    sqMat = diffMat**2
    sqDistance = sqMat.sum(axis = 1)
    distance = sqDistance**0.5
    kDistanceIndex = distance.argsort()
    '''   
    print type(kDistanceIndex)
    print kDistanceIndex.shape[0]
    print kDistanceIndex
    '''
    predication = 0
    classCount = {}
    for i in range(k):
        votelabel = labels[kDistanceIndex[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.items(),lambda x,y: cmp(x[1],y[1]),reverse = True)
    predication = sortedClassCount[0][0]
    if Ylabel == predication :
        return True
    else:
        return False
def pca(dataSet,kDomain):
    avgMat = (dataSet.mean(axis = 0))
    avgMat = avgMat.reshape((1,len(avgMat)))
    #print avgMat.shape
    diffMat = dataSet - np.tile(avgMat,(dataSet.shape[0],1))
    covMat = (diffMat.T.dot(diffMat))/(dataSet.shape[0])
    eigValue = np.linalg.eig(covMat)[0]
    #print eigValue.shape

    eigVector = np.linalg.eig(covMat)[1]
    #print eigVector.shape
    maxIndex = np.argsort(-eigValue)
    #print maxIndex.shape
    p = np.zeros((eigVector.shape[0],kDomain))
    for i in range(kDomain):#

        tmp = eigVector[:,maxIndex[i]:maxIndex[i]+1]

        #np.column_stack((p,tmp))
        p[:,i:i+1] = tmp
    #print p
    return dataSet.dot(p)
def folder5(dataSet,labels,k):
    Xtrain,Ytest,Xlabels,Ylabels = cross_validation.train_test_split(dataSet,labels,test_size = 0.1, random_state = 0)
    #print Ylabels.shape
    YtestSize = Ytest.shape[0]
    cnt = 0
    for i in range(YtestSize):
        if KNN(Ytest[i:i + 1, :], Ylabels[i], Xtrain, Xlabels, k=k):
            cnt = cnt + 1

    #print cnt
    return cnt / YtestSize

def kFinder(dataSet,labels):
    max = 0
    bestk = 0
    y = []
    for i in range(150):
        tmp = folder5(dataSet,labels,i + 1)
        y.append(tmp)
        if  tmp > max:

            max = tmp
            bestk = i

    plt.plot(y)
    plt.show()
    return bestk
if __name__ == '__main__':
    '''
    test for KNN
    '''
    '''
    a = np.array([[1.,1.1],
                [1.,1],
                [0.,0.],
                [0.,0.1]])
    x = [1,1,2,2]
    b = np.array([[2,2]])
    y = [1]
    print a[0][0]
    '''


    """
    test for pca
    """
    dataSet = pca(dataSet,6)
    '''
    a = np.array([[-1,-2],[-1,0],[0,0],[2,1],[0,1]])
    print pca(a,1) 
    '''


    '''
    test for folder5
    '''

    print folder5(dataSet,labels,10)
    

    '''
    test for kFinder
    '''
    '''
    
    '''


    print kFinder(dataSet,labels)







