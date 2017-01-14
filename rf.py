# -*- coding: utf-8 -*-
import numpy as np
import math
import sys
from datasets import Sample
              
## Hyperparameters
sys.setrecursionlimit(10000)
soF = 1 # number of features randomly chosen
MAXDEPTH = 10#math.inf

## Psuedo Code
'''
Precondition: A training set S := (x1, y1), . . . ,(xn, yn), features F, and number
of trees in forest B.
1 function RandomForest(S , F)
    2 H ← ∅
    3 for i ∈ 1, . . . , B do
        4 S(i) ← A bootstrap sample from S
        5 hi ← RandomizedTreeLearn(S(i), F)
        6 H ← H ∪ {hi}
    7 end for
    8 return H
9 end function

10 function RandomizedTreeLearn(S , F)
    11 At each node:
        12 f ← very small subset of F
        13 Split on best feature in f
    14 return The learned tree
15 end function
'''

def RandomForest(S, nbrTrees = 100, F=None, SF=True):
    """ Return the random forest trained on data S, with trees H[i] and out-of-bag data samples Sb[i] for the i-th tree."""
    if F == None:
        F = range(S[0].getNbrAttributes())
    B = nbrTrees                    # number of trees in forest
    H = []
    Sb = []
    Sset = set(S)
    for i in range(B):
        [S_i, Stst] = BootstrapSample(S, Sset)
        h_i = RandomizedTreeLearn(S_i, F, SF)
        H.append(h_i)
        Sb.append(Stst)
    return [H, Sb]

def RandomizedTreeLearn(S, F, singleFeature, maxdepth = MAXDEPTH): # here S is a subsample 
    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        allClass,theClass = allAnyClass(dataset)
        if allClass:
            return TreeLeaf(theClass)
        return RandomizedTreeLearn(dataset, attributes, singleFeature, maxdepth-1)

    default = mostCommon(S)
    if maxdepth < 1:
        return TreeLeaf(default)
    f = FeatureSubsample(F, singleFeature) # feature subsample
    [a, v, dataR, dataL] = bestFeature(S, f)
    attributesLeftL = F
    attributesLeftR = F
    #[dataR, dataL] = select(S, a, v)
    #dataL = select(S, a, v, False) #under
    #dataR = select(S, a, v, True) #over
    branches = [buildBranch(dataL, default, attributesLeftL),
                buildBranch(dataR, default, attributesLeftR)]
    return TreeNode(a, branches, default, v)

def BootstrapSample(S, Sset):
    soS = int(len(S) * 2/3)    # size of subsample
    train = np.random.choice(S, soS, replace = True).tolist()
    tst = Sset.difference(set(train))
    return [train, tst] 

def FeatureSubsample(F, singleFeature):
    soS = soF
    if not singleFeature:
        soS = int(math.log(len(F)+1,2))
    return np.random.choice(F, soS).tolist()

class TreeNode:
    def __init__(self, attribute, branches, default, value):
        self.attribute = attribute
        self.branches = branches
        self.default = default
        self.value = value
        
    def __repr__(self):
        accum = str(self.attribute) + '('
        for branch in self.branches:
            accum += str(branch)
        return accum + ')'
    
class TreeLeaf:
    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        return str(self.cvalue)

def getAttributeValues(S,attribute):
    values = []
    for x in S:
        val = x.getAttributeValue(attribute)
        if not val in values:
            values += [val]
    return sorted(values)
    
def bestFeature(S, features):
    if len(S) <= 1:
        return None
    top = None
    tover = None
    tunder = None
    for f in features:
        for v in getAttributeValues(S,f):
            [gain, sover, sunder] = averageGain(S, f, v)
            if top is None:
                top = (gain, f, v)
                tover = sover
                tunder = sunder
            elif top[0] < gain:
                top = (gain, f, v)
                tover = sover
                tunder = sunder
    #gains = [(averageGain(S, f, v), f, v) for f in features for v in getAttributeValues(S,f)]
    #return max(gains, key=lambda x: x[0])[1:]
    return [top[1], top[2], tover, tunder]
    
def mostCommon(S):
    classes = list(set([x.getClass() for x in S]))
    return classes[np.argmax([len([x for x in S if x.getClass()==c]) for c in classes])]

def averageGain(dataset, attribute, value):
    weighted = 0.0
    [subsetOver, subsetUnder] = select(dataset, attribute, value)
    weighted += entropy(subsetOver) * len(subsetOver)
    weighted += entropy(subsetUnder) * len(subsetUnder)
    return [entropy(dataset) - weighted/len(dataset), subsetOver, subsetUnder]


def entropy(dataset):
    "Calculate the entropy of a dataset"
    n = len(dataset)
    entropy = 0
    counts = dict()
    for x in dataset:
        cls = x.getClass()
        if cls not in counts:
            counts[cls] = 1
        else:
            counts[cls] += 1
    for v in counts.values():
        entropy -= float(v)/n * math.log(float(v)/n,2)
    return entropy
    
"""
def entropy(dataset):
    "Calculate the entropy of a dataset"
    classes = list(set([x.getClass() for x in dataset]))
    n = len(dataset)
    entropy = 0
    for c in classes:
        nclass = len([x for x in dataset if x.getClass() == c])
        if nclass == 0:
            continue
        else:
            entropy -= float(nclass)/n * math.log(float(nclass)/n,2)
    return entropy
"""

def select(dataset, attribute, value):
    #print('value', value)
    #print('attribute', attribute)
    #print(dataset[0].attribute[attribute])
    over = []
    under = []
    for x in dataset:
        if (x.getAttributeValue(attribute) >= value):
            over += [x]
        else:
            under += [x]
    return [over, under]

def allFromClass(dataset,c):
    "Check if all samples are from class c"
    return all([x.getClass() == c for x in dataset])

def allAnyClass(dataset):
    if len(dataset) == 0:
        return (False,0)
    "Check if all samples are from the same class"
    c = dataset[0].getClass()
    if allFromClass(dataset,c):
        return (True,c)
    return (False,0)
