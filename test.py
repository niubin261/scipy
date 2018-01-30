import numpy as np
#from scipy.optimize import minimize, fminbound, brent
#from scipy.special import jv
#from matplotlib import markers
#from matplotlib.pyplot import plot, show
#from pylab import *
#from sklearn import preprocessing
# def rosen(x):
#     """The Rosenbrock function"""
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
#
# x = np.linspace(0, 2 * np.pi, 30)
# wave = np.cos(x)
# transformed = np.fft.fft(wave)
# plot(transformed)
# show()
#
# a= np.zeros(1000)
# a[:100]=1
# b=np.fft.fft(a)
# plot(abs(b))
# show()
# print(res.x)
#
# for k in arange(0.5,5.5):
#      y = jv(k,x)
#      plot(x,y)
#      f = lambda x: -jv(k,x)
#      x_max = fminbound(f,0,6)
#      plot([x_max], [jv(k,x_max)],'ro')
#
# title('Different Bessel functions and their local maxima')
# show()
#
# def f(x):
#     return -np.exp(-(x-0.7)**2)
# x_min=brent(f)
# x_min=fminbound(f,1,2,xtol=0.000001,maxfun=1,disp=2)
# print x_min
#
# import numpy as np
# import urllib
# # url with dataset
# from sklearn import metrics
# from sklearn.svm import SVC
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# # download the file
# raw_data = urllib.urlopen(url)
# # load the CSV file as a numpy matrix
# dataset = np.loadtxt(raw_data, delimiter=",")
# # separate the data from the target attributes
# X = dataset[:,0:7]
# y = dataset[:,8]
#
# # fit a SVM model to the data
# model = SVC()
# model.fit(X, y)
# print(model)
# # make predictions
# expected = y
# predicted = model.predict(X)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))


# a = np.array((1,2,3))
# print a
# b = np.array((2,3,4))
# print b
# c=np.dstack((a,b))
# print c
# print c.shape
#
# phi=np.ndarray((2,1))
# print phi
# p=np.array((1,1))
# print p
# phi=np.dstack((phi,p))
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def test():
    a = np.ndarray([[1],[0],[2]])
    b = np.argsort(a)
    print b
def test2():
    x = {"hello":1,"world":0}
    y = "hello"


    z = sorted(x.items(), lambda x, y: cmp(x[1], y[1]))
    print z
    print x[y]
    print x.get(y,0)
def test3():
    a = np.array([[1.,1.1],
                [1.,1],
                [0.,0.],
                [0.,0.1]])
    x = [1,1,2,2]
    b = np.array([[2,2]])
    y = [1]
    print a[0][0]
def test4():
    a = np.array([[1,2,3],[4,5,6]])
    print a.shape
    b = a.mean(axis = 0)
    b = b.reshape((1,len(b)))
    print b.shape
    print b
    print type(b)
    print a.mean(axis = 1)
    c = a.sum(axis = 1)
    print c

    #d = np.array([1 2])

def test5():
    a = (1,2,3,4)

    for i in range(len(a)):
        print i
        print "hello"
def func():

    #return i
    pass
def test6():
    print sum([1,2,3,4,5])
def test7():
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[1,2,3]])
    print a[0]
    c = b*a[0]
    print c
def test8():
    a = np.zeros((2,2))
    b = np.mat([[1,2],[3,4]])

    for i in range(2):
        temp = b[:,i:i+1]
        print temp

        print temp.shape
        a[:,i:i+1] = temp
    print a
def test9():
    pgent = np.array([[0.13340885,0.37808]])
    x = np.argsort(-pgent,axis = 1)
    print x
def test10():
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.ones((2,3))
    print b[:,0:1]

    c = np.insert(a,3,b[:,0:1],axis=1)
    print c
def test11():
    xtrain = np.array([[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]])
    ytrain = np.array([[7], [9], [13], [17.5], [18]])
    ploy = preprocessing.PolynomialFeatures(2)
import matplotlib.pyplot as plt
def test12():
    x = np.array([[1,2,3,4]])
    #x.reshape((4,1))

    y = np.array([[5,0,9,11]])
    #y.reshape((4,1))
    x = x.tolist()
    y = y.tolist()
    print x
    print y
    plt.plot(x,y,'-o')
    x = [1,2,3,4]
    y = [2,5,7,8]
    plt.plot(x,y)
    plt.show()
def test13():
    results = np.zeros((10,10))
    for i in range(0,10,1):
        for j in range(0,10,1):
            results[i][j] = 1
    results[0][0] = 32
    from mpl_toolkits.mplot3d import Axes3D
    x = np.arange(0,10)
    y = np.arange(0,10)
    x,y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x,y,results)
    plt.show()


if __name__ == '__main__':

    test13()
    #i = 1
   # print "func = %d" %func()