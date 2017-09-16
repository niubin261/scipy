import numpy as np
from scipy.optimize import minimize, fminbound, brent
from scipy.special import jv
from matplotlib import markers
from matplotlib.pyplot import plot, show
from pylab import *
from sklearn import preprocessing
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})

x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
plot(transformed)
show()

a= np.zeros(1000)
a[:100]=1
b=np.fft.fft(a)
plot(abs(b))
show()
print(res.x)

for k in arange(0.5,5.5):
     y = jv(k,x)
     plot(x,y)
     f = lambda x: -jv(k,x)
     x_max = fminbound(f,0,6)
     plot([x_max], [jv(k,x_max)],'ro')

title('Different Bessel functions and their local maxima')
show()

def f(x):
    return -np.exp(-(x-0.7)**2)
x_min=brent(f)
x_min=fminbound(f,1,2,xtol=0.000001,maxfun=1,disp=2)
print x_min

import numpy as np
import urllib
# url with dataset
from sklearn import metrics
from sklearn.svm import SVC
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]

# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


