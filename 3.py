import numpy as np
from numpy import random
from scipy import constants
import matplotlib.pyplot as plt
import numpy.linalg as lg
num=7
x_origin=random.rand(25,100)
x_origin.sort(axis=0)
gauNoise=np.random.normal(0,0.1,size=(25,100))
t0=np.sin(2*constants.pi*x_origin)
t=t0+gauNoise
ln=np.arange(-10,0,1)
alpha=np.exp(ln)
sumy=np.zeros(100)
def polyweigth(x0,t,alpha,num):
    m=len(x0)
    print m
    phi=np.ndarray(m)
    for i in range(1,num+1,1):
        p=np.power(x0,i)
        phi=np.vstack((phi,p))
    e=np.eye(num+1)
    phiT=phi.T
    print phi.shape
    print phiT.shape
    A=alpha*e
    B=phi.dot(phiT)
    A=lg.inv(A+B)
    w=A.dot(phi.dot(t))
    return w

for k in range(0,10,1):
    plt.figure(k)

    x=np.arange(0,1,0.01)
    print x.shape
    for i in range(0,100,1):
        xx=np.ones(100)
        w=polyweigth(x_origin[:,i],t[:,i],alpha[k],num)
        for j in range(1,num+1,1):
            xj=np.power(x,j)
            xx=np.vstack((xx,xj))
        y=xx.T.dot(w)
        sumy=sumy+y
        plt.title(("ln(lambda)" ))
        plt.subplot(211)

        plt.plot(x,y)

    avey=sumy/len(x)
    plt.subplot(212)
    plt.plot(x,avey,x,np.sin(2*constants.pi*x))

plt.show()






















