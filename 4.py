import numpy as np
import matplotlib.pyplot as plt
def originalPerceptron(w0,x0,origin,eta):
    n=len(origin)
    w=w0
    print w
    print w.shape
    compare=origin[0]
    for i in range(0,n):
        if origin[i]==compare:
            origin[i]=1
        else:
            origin[i]=-1
    print origin

    print compare
    mark=1
    iteration=0
    while mark>0 :
        mark=n
        for i in range(0,n):
            if x0[i].dot(w)*origin[i]<=0:
                w=w+eta*origin[i]*(x[0].T)
            else:
                mark=mark-1

        iteration=iteration+1
        if iteration>500:
            break
    print w
    print w.shape
    w=np.reshape(w,(1,3))
    print w
    print w.shape

    return w
def dualPerceptron(x0,origin,eta):
    n=len(origin)
    compare=origin[1]
    for i in range(0,n):
        if origin[i]==compare:
            origin[i]=1
        else:
            origin[i]=-1
    mark=1

    a=np.zeros((1,n))
    print a
    print origin
    w=(a*origin).dot(x0)
    print w
    iteration=0
    while mark>0:
        mark=n
        for i in range(0,n):
            if origin[i]*x0[i].dot(w.T)<=0:
                a[0][i]=a[0][i]+eta
            else:
                mark=mark-1
            w=(a*origin).dot(x0)
        iteration=iteration+1
        if iteration>500:
            break
    print a
    print w
    print w.shape
    return w

x=np.array([[3, 3, 1], [4, 3, 1], [1, 1, 1]])
t0=np.array([1,1,0])
[m,n]=x.shape
w0=np.array([1,10,0])
w0=w0.T
wOriginal=originalPerceptron(w0, x, t0, 1)
wDual=dualPerceptron(x, t0, 1)
q=1
p=1
x01=np.zeros((m,n))
x02=np.zeros((m,n))
for i in range(0,m):
    if(t0[i]==t0[0]):
        x01[p][0]=x[i][0]
        x01[p][1]=x[i][1]
        p=p+1
    else:
        x02[q][0]=x[i][0]
        x02[q][2]=x[i][1]
        q=q+1

maxi=max(x[:0])
mini=min(x[:0])
x1=np.arange(mini-2,maxi+2+1,step=1)
x2d=-(wOriginal[0][0]*x1+wOriginal[0][2])/wOriginal[0][1]
x2o=-(wDual[0][0]*x1+wDual[0][2])/wDual[0][1]
plt.plot(x1,x2o,)





