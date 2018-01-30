from __future__ import division
import math

from sklearn.preprocessing import PolynomialFeatures
import re
import numpy as np
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return np.array(dataMat)
'''
linear regression
'''
from sklearn.linear_model import LinearRegression
def lineReg(xtrain,ctrain,k,xtest,ytest,ptest):
    ploy = PolynomialFeatures(k)
    xtrain = ploy.fit_transform(xtrain)
    xtest = ploy.transform(xtest)
    model = LinearRegression()
    model.fit(xtrain,ctrain)
    #print model.coef_.shape
    predictions = model.predict(xtest)
    #print predictions.shape
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i][0]*( 1 - ytest[i][0])
        else:
            prediction = 0
            errorRate += ptest[i][0]*(ytest[i][0])
    return errorRate
    #print errorRate
def kfoldLinearReg(x_train,x_test,y_train,y_test,k):
    ploy = PolynomialFeatures(k)
    x_train = ploy.fit_transform(x_train)
    x_test = ploy.fit_transform(x_test)
    model = LinearRegression()
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    cnt = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction != y_test[i]:
            cnt += 1
    print cnt
    return cnt / len(x_test)
from sklearn.linear_model import Lasso
def lossoReg(xtrain,ctrain,k,xtest,ytest,ptest,alpha):
    poly = PolynomialFeatures(k)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)
    model = Lasso(alpha = alpha)
    model.fit(xtrain,ctrain)
    predictions = model.predict(xtest)
    errorRate = 0
    tmp = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            tmp = ptest[i][0]* (1 - ytest[i][0])
            errorRate += tmp
        else:
            tmp = ptest[i][0]*(ytest[i][0])
            errorRate += tmp
    return errorRate
from sklearn.linear_model import Ridge
def redgeReg(xtrain,ctrain,k,xtest,ytest,ptest,alpha):
    #print xtest.shape
    poly = PolynomialFeatures(k)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)
    model = Ridge(alpha = alpha,solver = 'lsqr')
    model.fit(xtrain,ctrain)
   # print model.get_params(deep = True)
   # print model.coef_
   # print model.coef_.shape
    #print xtest.shape
    predictions = model.predict(xtest)
    #print predictions.shape
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i][0]*(1 - ytest[i][0])
        else:
            prediction = 0
            errorRate += ptest[i][0]*( ytest[i][0])
    #print errorRate
    return errorRate
from sklearn.linear_model import LogisticRegression
def logisticReg(xtrain,ctrain,k,xtest,ytest,ptest):
    poly = PolynomialFeatures(k)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)

    model = LogisticRegression(solver='liblinear')
    model.fit(xtrain,ctrain)
    predictions = model.predict(xtest)
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i][0]*(1 - ytest[i][0])
        else:
            prediction = 0
            errorRate += ptest[i][0]*(ytest[i][0])

    #print errorRate
    return errorRate
'''
using gradient descent 
'''
def myLogisticReg(xtrain,ctrain,k,xtest,ytest,ptest):
    poly = PolynomialFeatures(2)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)

    pass
def kmeans():
    pass
def lda():
    pass


if __name__ == '__main__':
    xtrain = loadDataSet('xtrain.txt')
    #print xtrain
   # print xtrain.shape
    ctrain = loadDataSet('ctrain.txt')
   # print ctrain.shape
    xtest = loadDataSet('xtest.txt')
   # print xtest.shape
    ytest = loadDataSet('c1test.txt')
   # print ytest.shape
    ptest = loadDataSet('ptest.txt')

    '''
    test for linear Reg
    '''

    import matplotlib.pyplot as plt
    results = []
    for i in range(1,50,1):

        result = lineReg(xtrain,ctrain,i,xtest,ytest,ptest)

        results.append(result)

    m = 1
    k = -1
    print results
    for i in range(len(results)):
        if results[i]<m:
            m = results[i]
            k = i
    print m
    print k

    plt.plot(results)
    plt.show()

    '''
    test for kfold
    '''

    from sklearn import cross_validation
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(xtrain,ctrain,test_size=0.1,random_state=0)
    print kfoldLinearReg(x_train,x_test,y_train,y_test,17)
    results = []
    for i in range(1,50,1):
        results.append(kfoldLinearReg(x_train,x_test,y_train,y_test,i))
    m = 1
    k = -1
    for i in range(len(results)):
        if results[i]<m:
            m = results[i]
            k = i
    print m
    print k
    import matplotlib.pyplot as plt
    plt.plot(results)
    plt.show()


    '''
    test for redge Reg
    '''
    results = np.zeros((5,50))
    m = 1
    k = -1
    l = -1
    for i in range(0,5,1):
        for j in range(0,50,1):
            tmp = (lossoReg(xtrain,ctrain,j + 1,xtest,ytest,ptest,i + 1))
            results[i][j] = tmp
            if m > tmp:
                m = tmp
                k = i
                l = j
    print m
    print k
    print l
    x = np.arange(0,50)
    y = np.arange(0,5)
    x,y = np.meshgrid(x,y)
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    print x.shape
    print y.shape
    print results.shape
    ax.plot_surface(x,y,results)
    plt.show()
    '''
    test for losso
    '''
    m = 1
    k = -1
    l = -1
    for i in range(1,50,1):
        for j in range(1,5,1):
            tmp = (lossoReg(xtrain,ctrain,i,xtest,ytest,ptest,j))
            if m > tmp:
                m = tmp
                k = i
                l = j
    print m
    print k
    print l
    '''
    test for logistic
    '''
    import matplotlib.pyplot as plt
    results = []
    for i in range(1,50,1):

        result = logisticReg(xtrain,ctrain,i,xtest,ytest,ptest)

        results.append(result)

    m = 1
    k = -1
    print results
    for i in range(len(results)):
        if results[i]<m:
            m = results[i]
            k = i
    print m
    print k

    plt.plot(results)
    plt.show()