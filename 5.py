from __future__ import division
import numpy as np

def naiveBayes(inputX,outputY,laplace):
    length=len(inputX)
    N1 = [[0,0,0], [0,0,0]]
    N2 = [[0,0,0], [0,0,0]]
    cnt1=0
    cnt_1=0
    for i in range(0,length,1):
        if(inputX[i][2]==1):
            cnt1=cnt1+1
        else:
            cnt_1=cnt_1+1
    for i in range(0,length,1):
        if(inputX[i][2]==1):
            if(inputX[i][0]==1):
                N1[0][0]=N1[0][0]+1
            if(inputX[i][0]==2):
                N1[0][1]=N1[0][1]+1
            if(inputX[i][0]==3):
                N1[0][2]=N1[0][2]+1
        else:
            if(inputX[i][0]==1):
                N1[1][0]=N1[1][0]+1
            if(inputX[i][0]==2):
                N1[1][1]=N1[1][1]+1
            if(inputX[i][0]==3):
                N1[1][2]=N1[1][2]+1

    for i in range(0,length,1):
        if(inputX[i][2]==1):
            if(inputX[i][1]=='s'):
                N2[0][0]=N2[0][0]+1
            if(inputX[i][1]=='m'):
                N2[0][1]=N2[0][1]+1
            if(inputX[i][1]=='l'):
                N2[0][2]=N2[0][2]+1
        else:
            if(inputX[i][1]=='s'):
                N2[1][0]=N2[1][0]+1
            if(inputX[i][1]=='m'):
                N2[1][1]=N2[1][1]+1
            if(inputX[i][1]=='l'):
                N2[1][2]=N2[1][2]+1

    N1=np.matrix(N1)
    N2=np.matrix(N2)
    N1=N1+laplace*np.ones((2,3))
    N2=N2+laplace*np.ones((2,3))

    py_1=(cnt_1+laplace)/(length+laplace*2)
    py1=(cnt1+laplace)/(length+laplace*2)


    if outputY[0]==1:
        py1*=(N1[0,0]+laplace)/(cnt1+laplace*3)
        py_1*=(N1[1,0]+laplace)/(cnt_1+laplace*3)
    if outputY[0]==2:
        py1 *= (N1[0,1] + laplace) / (cnt1 + laplace * 3)
        py_1 *= (N1[1,1] + laplace) / (cnt_1 + laplace * 3)
    if outputY[0]==3:
        py1 *= (N1[0,2] + laplace) / (cnt1 + laplace * 3)
        py_1 *= (N1[1,2] + laplace) / (cnt_1 + laplace * 3)



    if outputY[1]=='s':
        py1*=(N2[0,0]+laplace)/(cnt1+laplace*3)
        py_1*=(N2[1,0]+laplace)/(cnt_1+laplace*3)
    if outputY[1]=='m':
        py1 *= (N2[0,1] + laplace) / (cnt1 + laplace * 3)
        py_1 *= (N2[1,1] + laplace) / (cnt_1 + laplace * 3)
    if outputY[1]=='l':
        py1 *= (N2[0,2] + laplace) / (cnt1 + laplace * 3)
        py_1 *= (N2[1,2] + laplace) / (cnt_1 + laplace * 3)
    print py_1
    print py1

    return 1 if py1>py_1 else -1

if __name__ == '__main__':
   inputX=[[1,'s',-1],[1,'m',-1],[1,'m',1],[1,'s',1],[1,'s',-1],[2,'s',-1],[2,'m',-1],[2,'m',1],[2,'l',1],[2,'l',1],[3,'l',1]]
   outputY=[1,'m']
   print naiveBayes(inputX,outputY,0)

   print naiveBayes(inputX,outputY,1)

