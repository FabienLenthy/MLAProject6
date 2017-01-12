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

def chooseData(data):
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
        return [Sample(data[x][-1],[data[x][i] for i in range(len(data[x])-1)],x) for x in range(len(data))]

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

