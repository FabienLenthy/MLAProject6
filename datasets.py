# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# TODO: lighter dataset representation? np.array?
class Sample:
    "Represenation of data samples"
    def __init__(self, xclass, values, identity, attributes=None):
        self.xclass = xclass
        if attributes == None:
            self.attribute = values
        else:
            self.attribute = dict(zip(attributes, values))
        self.identity = identity
        
    def getClass(self):
        return self.xclass
    
    def getAttributes(self):
        return self.attribute
    
    def getNbrAttributes(self):
        return len(self.attribute)
    
    def getAttributeValue(self,attribute):
        return self.attribute[attribute]
    
    def getIdentity(self):
        return self.identity

CancerAttributes = (
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size',
                'Bare Nuclei',
                'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses'
              )

def waveform(nbr):
    NUMBER_OF_ATTRIBUTES = 21
    NUMBER_OF_CLASSES = 3
    data = []
    
    h = np.zeros((NUMBER_OF_CLASSES,NUMBER_OF_ATTRIBUTES))
    h[0][1:7] = range(1,7)
    h[0][7:12] = range(1,6)[::-1]
    h[1][9:20] = h[0][1:12]
    h[2][5:16] = h[0][1:12]
    
    for i in range(nbr):
        cls = np.random.randint(3)
        choice = [int(cls==2),2-int(cls==0)]
        m = float(np.random.randint(1000))/1000.0
        randValue = 0
        data += [[m*h[choice[0]][i] + (1-m)*h[choice[1]][i] + randValue for i in range(NUMBER_OF_ATTRIBUTES)] + [cls]]
        
    return data

def twonorm(nbr):
    dim=20
    a = 2.0/20**0.5
    m = [a,-a]
    variance = 1
    data = []
    
    for i in range(nbr):
        cls = np.random.randint(2)
        data += [[np.random.normal(m[cls],variance) for _ in range(dim)] + [cls]]
    
    return data

def threenorm(nbr):
    dim=20
    a = 2.0/20**0.5
    means = [[a]*dim,[-a]*dim,[a,-a]*int(dim/2)]
    variance = 1
    data = []
    
    for i in range(nbr):
        cls = np.random.randint(2)
        mean = []
        if cls == 0:
            mean = means[np.random.randint(1)]
        else:
            mean = means[2]
        data += [[np.random.normal(mean[j],variance) for j in range(dim)] + [cls]]
    
    return data
    
def ringnorm(nbr):
    dim=20
    a = 2.0/20**0.5
    m = [0,a]
    variance = [4,1]
    data = []
    
    for i in range(nbr):
        cls = np.random.randint(2)
        data += [[np.random.normal(m[cls],variance[cls]) for _ in range(dim)] + [cls]]
    
    return data

def chooseData(data,nbr=3300):
    prefix = 'datasets/'
    if data == "glass":
        data =np.genfromtxt(prefix + 'glass.data',dtype=None,delimiter=",")
        return [Sample(x[-1],[x[i] for i in range(1,len(x)-1)],x[0]) for x in data]

    elif data == "ecoli":
        data = np.genfromtxt(prefix + 'ecoli.data',dtype=None)
        return [Sample(x[-1],[x[i] for i in range(1,len(x)-1)],x[0]) for x in data]

    elif data == "diabetes":
        data = np.genfromtxt(prefix + 'pima-indians-diabetes.data',dtype=None,delimiter=",")
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "sonar":
        data = np.genfromtxt(prefix + 'sonar.all-data',dtype=None,delimiter=",")
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "vowel":
        data = np.genfromtxt(prefix + 'vowel-context.data',dtype=None)
        return [Sample(data[x][-1],[data[x][i] for i in range(3, len(data[x])-1)],x) for x in range(len(data))]

    elif data == "ionosphere":
        data = np.genfromtxt(prefix + 'ionosphere.data',dtype=None, delimiter=',')
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "vehicle":
        data = np.genfromtxt(prefix + 'vehicle.data',dtype=None, delimiter=',')
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "german":
        data = np.genfromtxt(prefix + 'german.data-numeric',dtype=None)
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "image":
        data = np.genfromtxt(prefix + 'segment.dat',dtype=None)
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

    elif data == "cancer":
        data = np.genfromtxt(prefix + 'breast-cancer-wisconsin.data',delimiter=",",dtype=int)
        return [Sample(x[-1],[x[i] for i in range(1,len(x)-1)],x[0]) for x in data]

    elif data == "votes":
        data = np.genfromtxt(prefix + "house-votes-84.data",dtype=str,delimiter=',')
        #data = [list(x) for x in data if not any([y == '?' for y in x])]
        data = [[int(y == 'y' or y == 'democrat') - int(y == '?') for y in x] for x in data] # to numerical values
        return [Sample(data[x][0],[data[x][i] for i in range(1,len(data[x]))],x) for x in range(len(data))]

    elif data == "liver":
        data = np.genfromtxt(prefix + "bupa.data",dtype=None,delimiter=',')
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "letter":
        data = np.genfromtxt(prefix + "letter-recognition.data",dtype=None,delimiter=',')
        return [Sample(data[x][0],[data[x][i] for i in range(1,len(data[x]))],x) for x in range(len(data))]
    
    elif data == "satellite":
        data = np.genfromtxt(prefix + "sat.trn",dtype=int)
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "satellite test":
        data = np.genfromtxt(prefix + "sat.tst",dtype=int)
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "soybean":
        data1 = np.genfromtxt(prefix + 'soybean-large.data',dtype=str,delimiter=',')
        data2 = np.genfromtxt(prefix + 'soybean-large.test',dtype=str,delimiter=',')
        data1 = [Sample(data1[x][0],[int(data1[x][i]) for i in range(1,len(data1[x]))],x) for x in range(len(data1)) 
                 if not any(y=='?' for y in data1[x])]
        data2 = [Sample(data2[x][0],[int(data2[x][i]) for i in range(1,len(data2[x]))],x) for x in range(len(data2)) 
                 if not any(y=='?' for y in data2[x])]
        return np.concatenate((data1,data2))
    
    elif data == "zip-code":
        data = np.genfromtxt(prefix + 'semeion.data',dtype=None)
        return [Sample([data[x][i] for i in range(len(data[x])-10,len(data[x]))].index(1),
                       [data[x][i] for i in range(len(data[x])-10)],x) for x in range(len(data))]
    
    elif data == "waveform":
        data = waveform(nbr)
        return [Sample(int(data[x][-1]),[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "twonorm":
        data = twonorm(nbr)
        return [Sample(int(data[x][-1]),[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "threenorm":
        data = threenorm(nbr)
        return [Sample(int(data[x][-1]),[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "ringnorm":
        data = ringnorm(nbr)
        return [Sample(int(data[x][-1]),[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "housing":
        data = np.genfromtxt(prefix + "housing.data",dtype=None)
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "ozone":
        data = np.genfromtxt(prefix + 'LAozone.data',dtype=float,delimiter=',')[1:]
        return [Sample(x[0],[x[i] for i in range(1,len(x)-1)],x[-1]) for x in data]
    
    elif data == "servo":
        data = np.genfromtxt(prefix + 'servo.data',dtype=None,delimiter=',')
        data = [[ord(data[x][0].decode('utf-8'))-65,ord(data[x][1].decode('utf-8'))-65] +
                [data[x][i] for i in range(2,len(data[x]))] for x in range(len(data))]
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
    
    elif data == "abalone":
        data = np.genfromtxt(prefix + 'abalone.data',dtype=None,delimiter=',')
        data = [[2*int(data[x][0].decode('utf-8')=='F')+int(data[x][0].decode('utf-8')=='M')]+
                [data[x][i] for i in range(1,len(data[x]))] for x in range(len(data))]
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]
