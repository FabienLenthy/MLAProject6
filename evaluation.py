# -*- coding: utf-8 -*-
from datasets import chooseData
from rf import RandomForest, TreeLeaf
import numpy as np
import time

def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[int(sample.getAttributeValue(tree.attribute)>=tree.value)], sample)

def classifyForest(forest, sample):
    "Classify a sample using the given decition tree"
    classifications = []
    for tree in forest:
        classifications += [classify(tree, sample)]
    counts = [(c,classifications.count(c)) for c in set(classifications)]
    return sorted(counts,key=lambda x: x[1])[-1][0]
    

def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.getClass():
            correct += 1
    return float(correct)/len(testdata)

def checkForest(forest, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classifyForest(forest, x) == x.getClass():
            correct += 1
    return float(correct)/len(testdata)

ITERATE = 10
# TODO: Try less iterations (3) and report variance, too.
# TODO: Save the reported data into a data structure.
# TODO: Report as a latex table from the data structure.
# TODO: Pickle the forests! 

def checkOnData(data):
    t0 = time.time()
    S = chooseData(data)
    N = len(S)
    cumulErrorSelection = 0
    cumulErrorF1 = 0
    cumulErrorF2 = 0
    cumulSpecificError = 0
    for i in range(ITERATE):
        nbrTrees = 100  # TODO: This number is dataset dependent! (zip-code 200)
        np.random.shuffle(S)
        train, test = S[:int(0.9*N)],S[int(0.9*N):]
        H, S_OOB = RandomForest(train, nbrTrees) # Modified to return OOB samples
        H2, S_OOB2 = RandomForest(train, nbrTrees, SF=False) # Modified to return OOB samples
        # Single tree: This should be out-of-bag, not test. Also the best one from H or H2.
        cumulSpecificError += sum([1-check(h,test) for h in H])/nbrTrees    
        errorF1 = 1-checkForest(H,test)
        errorF2 = 1-checkForest(H2,test)
        # Choose by means of the lowest out-of-bag error. Report the test error.
        cumulErrorSelection += min(errorF1,errorF2)  
        cumulErrorF1 += errorF1     # OK.
        
    finalErrorF1 = cumulErrorF1/ITERATE
    finalErrorSelection = cumulErrorSelection/ITERATE
    finalSpecificError = cumulSpecificError/ITERATE
    
    print("Data :", data)
    print("Number of input :", str(S[0].getNbrAttributes()))
    print("Number of data point :", str(len(S)))
    print("Error rate with selection :", str(finalErrorSelection))
    print("Error rate with single input :", str(finalErrorF1))
    print("Error rate with individual trees :", str(finalSpecificError))
    print('Execution time: {} seconds.'.format(time.time() - t0))
    print()
    
if __name__=="__main__":
    checkOnData("glass")
    checkOnData("cancer")
    checkOnData("diabetes")
    checkOnData("sonar")
    checkOnData("vowel")
    checkOnData("ionosphere")
    checkOnData("vehicle")
    checkOnData("german")
    checkOnData("image")
    checkOnData("ecoli")
    checkOnData("votes")
    checkOnData("liver")
