import numpy as np
from scipy.optimize import minimize, fminbound, brent
from scipy.special import jv
from matplotlib import markers
from matplotlib.pyplot import plot, show
from pylab import *

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
print x_min