# -*- coding: utf-8 -*-
from datasets import chooseData
from rf import RandomForest, TreeLeaf
import numpy as np
import time
import pickle

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

def metricsRI(oobForest1, oobForest2, oobTree1, oobTree2, errors1, errors2):
    """ Compute selection, singleInput and onetree metrics for a set of N forests."""
    N = len(oobForest1)
    selection = np.zeros(N)
    singleInput = np.zeros(N)
    onetree = np.zeros(N)
    
    for i in range(N):
        flag = 0
        # 'single input': test error, just 1 random feature to grow the trees
        singleInput[i] = errors1[i]
        
        # 'one tree': out-of-bag, averaged over individual trees, for the best setting (single or selection)
        onetree[i] = oobTree1[i]
        
        # The 'selection' result: 
        #   Choose the best forest based on the lowest out-of-bag error of that forest.
        if(oobForest1[i] < oobForest2[i]):
            selection[i] = errors1[i]
        else:
            selection[i] = errors2[i]
            if(selection[i] < singleInput[i]):
                onetree[i] = oobTree2[i]
    
    return [selection, singleInput, onetree]
        
def reportRI(selection, singleInput, onetree, data, S, runtime):
    mSel = np.mean(selection)
    mSingle = np.mean(singleInput)
    mOne = np.mean(onetree)
    stdSel = np.std(selection)
    stdSingle = np.std(singleInput)
    stdOne = np.std(onetree)
    print("Data :", data)
    print("Number of input :", str(S[0].getNbrAttributes()))
    print("Number of data point :", str(len(S)))
    print("Error rate with selection : {} +- {}".format(mSel, stdSel))
    print("Error rate with single input : {} +- {}".format(mSingle, stdSingle))
    print("Error rate with individual trees :{} +- {}".format(mOne, stdOne))
    print('Execution time: {} seconds.'.format(runtime))
    print()
    

def outOfBagError(H, Sb, train, param = 'forest'):
    N = len(H)
    oob = 0.0
    if param == 'tree':
        for i in range(N):
            oob += 1-check(H[i], list(Sb[i]))
        return oob/N
    elif param == 'forest':
        for s in train:
            Hm = []
            # Build a smaller forest and classify with it.
            for i in range(N):
                if s in Sb[i]:
                    Hm.append(H[i])
            assert(len(Hm) > 2) # This might fail but very rarely.
            oob += 1-checkForest(Hm, [s])
        return oob/len(train)
    else:
        raise('outOfBagError: unexpected parameter ' + param)
        return None
    
    
def save(ar, fname):
    f = open(fname, 'wb')
    pickle.dump(ar, f)
    f.close()
    
def load(fname):
    f = open(fname, 'rb')
    ar = pickle.load(f)
    f.close()
    return ar

ITERATE = 3
# We 'should' run 100 iterations over everything on the first 10 'small' datasets like in the paper! :-)
# TODO: Report as a latex table from the data structure.

def checkOnData(data, training = True, evaluating = True):
    t0 = time.time()
    modelPrefix = 'models/RI/'
    resultPrefix = 'results/RI/'
    S = chooseData(data)
    if data == "satellite":
        train = S
        test = chooseData("satellite test")
    N = len(S)
    # Test and out-of-bag errors.
    errors1 = np.zeros(ITERATE)
    errors2 = np.zeros(ITERATE)
    oobTree1 = np.zeros(ITERATE)
    oobTree2 = np.zeros(ITERATE)
    oobForest1 = np.zeros(ITERATE)
    oobForest2 = np.zeros(ITERATE)
    for i in range(ITERATE):
        nbrTrees = 100  # TODO: This number is dataset dependent! (zip-code 200)
        if training:
            if not data == "satellite":
                np.random.shuffle(S)
                train, test = S[:int(0.9*N)],S[int(0.9*N):]
            [H, Sb] = RandomForest(train, nbrTrees)
            [H2, Sb2] = RandomForest(train, nbrTrees, SF=False)
            save([[H, Sb, H2, Sb2], train, test], modelPrefix + data + 'Forests' + str(i) + '.pkl') 
        elif evaluating:
            [[H, Sb, H2, Sb2], train, test] = load(modelPrefix + data + 'Forests' + str(i) + '.pkl')
            nbrTrees = len(H)
            
        if evaluating:
            oobForest1[i] = outOfBagError(H, Sb, train, 'forest')
            oobForest2[i] = outOfBagError(H2, Sb2, train, 'forest')
            oobTree1[i] = outOfBagError(H, Sb, train, 'tree')
            oobTree2[i] = outOfBagError(H2, Sb2, train, 'tree')
            errors1[i] =  1-checkForest(H,test)
            errors2[i] =  1-checkForest(H2,test)
    runtime = time.time() - t0
    if not evaluating:
        [oobForest1, oobForest2, oobTree1, oobTree2, errors1, errors2, runtime] = load(resultPrefix + data + 'Results.pkl')
    else:
        save([oobForest1, oobForest2, oobTree1, oobTree2, errors1, errors2, runtime], resultPrefix + data + 'Results.pkl') 
    
    [selection, singleInput, onetree] = metricsRI(oobForest1, oobForest2, oobTree1, oobTree2, errors1, errors2)
    reportRI(selection, singleInput, onetree, data, S, runtime)
    
    
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
    checkOnData("letter")
    checkOnData("satellite")
