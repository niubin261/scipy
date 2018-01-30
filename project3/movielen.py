
from __future__ import division
import numpy as np
import datetime
import pandas as pd
from scipy import io as io
def perprocessData():
    movie = io.loadmat('movie.mat')
    #print movie
    rating = io.loadmat('rating.mat')
    #print rating
    user = io.loadmat('user.mat')
    movieMat = movie['movie']
    ratingMat = rating['rating']

    userMat0 = user['user0']
    userMat1 = user['user1']
    userMat2 = user['user2']
    userMat3 = user['user3']
    userMat4 = user['user4']
    userMat5 = user['user5']
    userMat6 = user['user6']
    userMat7 = user['user7']
    userMat8 = user['user8']
    userMat9 = user['user9']
    print userMat0
    print movieMat.shape
    print ratingMat.shape
    print userMat9.shape
    return movieMat,ratingMat,\
           userMat0,userMat1,userMat2,userMat3,userMat4,userMat5,userMat6,userMat7,userMat8,userMat9

def predict(movie,rating,usrTrain,usrTest):
    lm = len(movie)
    lu = len(usrTrain)#list
    lr = len(rating)
    pm14 = np.zeros((lm,14))
    pop2 = np.zeros((1,2))
    tmp = usrTrain[:,1:2]
    '''
    user num of different age and gender
    '''
    cnt = 0
    lu = len(usrTrain)
    for i in range(lu):
        if tmp[i][0] == 0:
           cnt = cnt +1
    pop2[0][0] = cnt
    pop2[0][1] = lu - cnt
    pop2 = pop2 / lu
    # pop2[0][0] = pop2[0][0]/len(usrTrain)
    # pop2[0][1] = 1 - pop2[0][0]
    pop7 = np.zeros((1,7))
    #print usrTrain
    for i in range(7):
        cnt = 0
        for j in range(lu):

            if usrTrain[j][2] == i + 1:
               cnt = cnt +1

        pop7[0][i] = cnt
    pop7 = pop7/lu
    #print pop7

    '''
    Relationship with movies
    movie,usr,rating --> arrary
    '''

    def find(data,key):
        ret = []
        length = len(data)
        for i in range(length):
            if data[i][0] == key:
                return i
        return
    for i in range(lr):
        su = find(usrTrain[:,0:1],rating[i][0])
        sm = find(movie[:,0:1],rating[i][1])
        if su is not None :
            gen_tmp = usrTrain[su][1]
            age_tmp = usrTrain[su][2]
            if sm is not None:
                pm14[sm][7*gen_tmp + age_tmp - 1 ] = pm14[sm][7*gen_tmp + age_tmp - 1] + 1

    pgen = np.zeros((lm,2))#gender
    page = np.zeros((lm,7))#age
    for i in range(lm):
        #print pm14[i][0]
        pgen [i][0] = sum([pm14[i][0],pm14[i][1],pm14[i][2],pm14[i][3],pm14[i][4],pm14[i][5],pm14[i][6]])
        pgen [i][1] = sum([pm14[i][7],pm14[i][8],pm14[i][9],pm14[i][10],pm14[i][11],pm14[i][12],pm14[i][13]])

        for j in range(7):
            page[i][j] = sum([pm14[i][j],pm14[i][j+7]])
    '''
    p(movie|age)
    p(movie|gen)
    '''
    for i in range(lm):
        #print pm14[i][0]
        tmp = pgen[i][0] + pgen[i][1]
        pgen[i][0] = pgen[i][0]/tmp
        pgen[i][1] = pgen[i][1]/tmp
        tmp = sum([page[i][0], page[i][1], page[i][2], page[i][3], page[i][4], page[i][5], page[i][6]])
        for j in range(7):

            page[i][j] = page[i][j]/tmp

    '''
    test
    '''
    start = datetime.datetime.now()
    errorage = 0
    errorgen = 0
    cnt = 0
    pgent = np.ones((1, 2))
    paget = np.ones((1, 7))
    lt = len(usrTest)
    uidTest = usrTest[:,0:1]
    gen = np.zeros((lt,1))
    age = np.zeros((lt,1))
    for i in range(lr):
        sut = find(uidTest[:,0:1],rating[i][0])
        smt = find(movie[:,0:1],rating[i][1])
        if sut is not None:
            gent_tmp = usrTest[sut][1]
            aget_tmp = usrTest[sut][2]
            if smt is not None:
                cnt += 1
                pgent = pop2*pgen[sut]
                paget = pop7*page[sut]
                y = np.argsort(-paget,axis = 1)
                x = np.argsort(-pgent,axis = 1)
                genp = x[0][0]
                agep = y[0][0] + 1
                if genp != gent_tmp:
                   # print genp
                   # print gent_tmp
                    errorgen +=  1
                if agep != aget_tmp:
                    errorage += abs(aget_tmp - agep)
    end = datetime.datetime.now()
    print cnt
    print lt
    print errorgen
    print errorage
    errorageRate = errorage/cnt
    errorgenRate = errorgen/cnt
    print errorage/lt
    print errorgen/lt
    print (end - start).seconds
    return errorageRate,errorgenRate
    pass
if __name__ == '__main__':
    '''
    test for preprocessing
    '''
    movie,rating,\
    usr0,usr1,usr2,usr3,usr4,usr5,usr6,usr7,usr8,usr9 = perprocessData()
    usrTrain = np.concatenate((usr0,usr1,usr2,usr3,usr4,usr5,usr6,usr7,usr8),axis = 0)
    print predict(movie,rating,usrTrain,usr9)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr3, usr4, usr5, usr6, usr7, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr8)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr3, usr4, usr5, usr6, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr7)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr3, usr4, usr5, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr6)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr3, usr4, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr5)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr3, usr5, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr4)
    usrTrain = np.concatenate((usr0, usr1, usr2, usr4, usr5, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr3)
    usrTrain = np.concatenate((usr0, usr1, usr3, usr4, usr5, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr2)
    usrTrain = np.concatenate((usr0, usr2, usr3, usr4, usr5, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr1)
    usrTrain = np.concatenate((usr1, usr2, usr3, usr4, usr5, usr6, usr7, usr8, usr9), axis=0)
    print predict(movie, rating, usrTrain, usr0)

    '''
    test for predict
    '''
    #movie = np.array([[1], [2], [3] [4], [5]])




