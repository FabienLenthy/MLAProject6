# -*- coding: utf-8 -*-
from datasets import chooseData
from rf import RandomForest, TreeLeaf
import numpy as np
import time
import pickle
import os

def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[int(sample.getAttributeValue(tree.attribute)>=tree.value)], sample)

def classifyForest(forest, sample, SOOB=None):
    "Classify a sample using the given decition tree"
    classifications = []
    for i in range(len(forest)):
        if SOOB == None or sample in SOOB[i]:   # TODO: This is not OOB forest classification! (among others, SOOB (almost) always empty)
            classifications += [classify(forest[i], sample)]
    if classifications == []:
        return None
    counts = [(c,classifications.count(c)) for c in set(classifications)]
    return sorted(counts,key=lambda x: x[1])[-1][0]
    

def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.getClass():
            correct += 1
    return float(correct)/len(testdata)

def checkForest(forest, testdata, SOOB=None):
    "Measure fraction of correctly classified samples"
    correct = 0
    N = len(testdata)
    
    for x in testdata:
        cls = classifyForest(forest, x, SOOB=SOOB)
        if cls == None:
            N -= 1
        elif cls == x.getClass():
            correct += 1
    if N == 0:  # TODO: Very likely always the case when doing OOB and that is wrong.
        return 1
    return float(correct)/N

# FOR OOB MARGIN
def OOBmarginTree(tree, testdata):
    "Calculates the OOB margin on tree basis - the strength of 'raw margin'"
    # TODO: This is not the raw margin,  it is more similar to check(tree, testdata).
    # raw margin: can be only -1, 0 or 1; it is computed only for one data point and one tree
    correct = 0
    incorrect = 0
    for x in testdata:
        if classify(tree, x) == x.getClass():
            correct += 1
        else:
            incorrect += 1
    ans = correct - incorrect
    return ans/100

def OOBmarginForest(forest, testdata, SOOB=None):
    "Calculates the OOB margin on forest basis - the strength of mean margin"
    # TODO: This is not oob forest margin.
    correct = 0
    incorrect = 0
    for x in testdata:
        cls = classifyForest(forest, x, SOOB=SOOB)
        if cls == None:
            continue
        elif cls == x.getClass():
            correct += 1
        else:
            incorrect += 1
    #ans = correct - incorrect
    ans = correct
    return ans/(len(forest))

def meanCorr(forest, testdata):
    "Calculates mean correlation from raw margins"
    # TODO: this is not mean correlation. 
    raw_vect = []
    for tree in forest:
        raw_str = OOBmarginTree(tree, testdata)
        raw_vect.append(raw_str)

    raw_mean = sum(raw_vect)/len(raw_vect) #mr(X,Y)
    raw_var = 0
    for i in raw_vect:
        raw_var += (i-raw_mean)**2
    raw_var = (raw_var/(len(raw_vect) - 1))**0.5
    mr2 = raw_mean*raw_mean
    mean_corr = raw_var/mr2
    return mean_corr


def raw_margin(datapoint, forest):
    "Calculates the margin of a data point, using expectation of raw_margin"
    x = datapoint 

    correct = 0
    incorrect = 0
    incorr_list = []
    incorr_count = []

    "In this loop we check which class is the 'best' among the wrong classes"
    for tree in forest:
        cls = classify(tree, datapoint)
        if cls == x.getClass():
            correct += 1
        else:
            incorrect += 1
            if cls not in incorr_list:
                incorr_list.append(cls)
                incorr_count.append(1)
            elif cls in incorr_list:
                incorr_count[incorr_list.index(cls)] += 1

    #hat(j) - raw margin function
    sum_ind_corr = correct

    if incorr_count == []: #We could have no incorrect classifications
        sum_ind_j = 0
    else:
        sum_ind_j = max(incorr_count)

    rmg = sum_ind_corr - sum_ind_j
                
    return rmg

def margin(datapoint, forest):
    "Calculates the margin of a data point, using expectation of raw_margin"
    x = datapoint 

    correct = 0
    incorrect = 0
    incorr_list = []
    incorr_count = []

    "In this loop we check which class is the 'best' among the wrong classes"
    for tree in forest:
        cls = classify(tree, datapoint)
        if cls == x.getClass():
            correct += 1
        else:
            incorrect += 1
            if cls not in incorr_list:
                incorr_list.append(cls)
                incorr_count.append(1)
            elif cls in incorr_list:
                incorr_count[incorr_list.index(cls)] += 1

    #hat(j) - raw margin function

    sum_ind_corr = correct
    if incorr_count == []: #We could have no incorrect classifications
        sum_ind_j = 0 
    else:
        sum_ind_j = max(incorr_count)

    rmg = sum_ind_corr - sum_ind_j

    "Expectation of raw margin of the data set given a tree classifier"
    exp_rmg = rmg /(correct + incorrect)
    mr = exp_rmg
    return mr

def strength(dataset, forest):
    "Calculates the strength"
    N = len(dataset)
    sum_mr = 0
    for x in dataset:
        mr = margin(x, forest)
        sum_mr += mr

    #Equal probability of datapoints (Univariate)
    #Therefore strength is same as mean of mr
    strength = sum_mr / N
    
    return strength

def varMR(dataset, forest):
    "Calculates variance of margin"
    N = len(dataset)
    mr_list = []
    for x in dataset:
        mr_list.append(margin(x, forest))
        
    mean_mr = sum(mr_list) / N

    var_mr = 0
    for i in mr_list:
        var_mr += (i-mean_mr)**2
    var_mr = var_mr/(N-1)

    return var_mr

def stdMR(dataset, forest):
    "Calculates standard deviation of margin"
    var = varMR(dataset, forest)
    stddev = var**0.5 #Variance is never negative, no worry about root of negative
    return stddev

def stdRMG(datapoint, forest):
    "Calculates the standard deviation for raw marginal"
    x = datapoint 

    correct = 0
    incorrect = 0
    corr_list = []
    corr_count = []
    incorr_list = []
    incorr_count = []
    
    for tree in forest:
        cls = classify(tree, datapoint)
        if cls == x.getClass():
            correct += 1
            if cls not in corr_list:
                corr_list.append(cls)
                corr_count.append(1)
            elif cls in corr_list:
                corr_count[corr_list.index(cls)] += 1
        else:
            incorrect += 1
            if cls not in incorr_list:
                incorr_list.append(cls)
                incorr_count.append(1)
            elif cls in incorr_list:
                incorr_count[incorr_list.index(cls)] += 1

    rmg_list = []
    if incorr_count == []:
        sum_rmg = 0
        for tree in forest:
            ind_corr = 0
            ind_argMax = 0
            cls = classify(tree,x)
            if cls == x.getClass():
                ind_corr = 1
            rmg = ind_corr - ind_argMax
            rmg_list.append(rmg)

    else:
        #hat(j) - raw margin function
        argMaxWrong_class = incorr_list[incorr_count.index(max(incorr_count))]

        #Raw marginal of data point
        sum_rmg = 0
        for tree in forest:
            ind_corr = 0
            ind_argMax = 0
            cls = classify(tree,x)
            if cls == x.getClass():
                ind_corr = 1
            elif cls == argMaxWrong_class:
                ind_argMax = 1

            rmg = ind_corr - ind_argMax
            rmg_list.append(rmg)
            
            sum_rmg += rmg

    mean_rmg = sum(rmg_list) / len(rmg_list)

    #Variance of rmg
    var_rmg = 0
    for i in rmg_list:
        var_rmg += (i-mean_rmg)**2
    var_rmg = var_rmg / (len(rmg_list) - 1)

    #Standard deviation of rmg
    stddev_rmg = var_rmg**0.5

    return stddev_rmg

def E_stdRMG(dataset, forest):
    "Calculates the expected value of the standard deviation of the raw marginals"
    stdrmg_list = []
    for x in dataset:
        stdrmg = stdRMG(x, forest)
        stdrmg_list.append(stdrmg)

    E_stdrmg = sum(stdrmg_list) / len(stdrmg_list)
    return E_stdrmg

# Mean correlation
def meanCorr(dataset, forest, forest2):
    "Calculates the mean correlation"
    var = varMR(dataset, forest)
    E_stdrmg = E_stdRMG(dataset, forest)
    E_stdrmg_prime = E_stdRMG(dataset, forest2)

    corrCoefficient = var / (E_stdrmg*E_stdrmg_prime)
    return corrCoefficient
    



"""
TODO:
    Check Q(x,j) as estimate for p(h(x,O) =j) ?
    Should we calculate all the above with estimates instead?
    Or maybe we should compare the estimates with true values?
"""

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

def latexRI(data, mSel, mSingle, mOne, stdSel, stdSingle, stdOne, fname):
    line = '{} & - & ${:.1f} \\pm {:.1f}$ & ${:.1f} \\pm {:.1f}$ & ${:.1f} \\pm {:.1f}$ \\\\ \n'.format(data, 
        mSel, stdSel, mSingle, stdSingle, mOne, stdOne)
    with open(fname, 'a') as f:
        f.write(line)
        
def reportRI(selection, singleInput, onetree, data, S, runtime, fname):
    mSel = np.mean(selection)*100.0
    mSingle = np.mean(singleInput)*100.0
    mOne = np.mean(onetree)*100.0
    stdSel = np.std(selection)*100.0
    stdSingle = np.std(singleInput)*100.0
    stdOne = np.std(onetree)*100.0
    latexRI(data, mSel, mSingle, mOne, stdSel, stdSingle, stdOne, fname)
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

def NoiseAdder(S):
    allClasses = [o.xclass for o in S]
    Classes = set(allClasses)
    print(Classes)
    N = len(S)
    np.random.shuffle(S)
    X1, X2 = S[:int(0.9*N)], S[int(0.9*N):]
    for o in X2:
        xx = o.xclass
        otherClasses = [c for c in Classes if c != o.xclass]
        np.random.shuffle(otherClasses)
        o.xclass = otherClasses[0]
    S = X1 + X2
    np.random.shuffle(S)
    return S

ITERATE = 3
latexTableFile = 'results/table.tex'
# We 'should' run 100 iterations over everything on the first 10 'small' datasets like in the paper! :-)

def checkOnData(data, training = False, evaluating = False, addNoise = False, simulatedSize = 3300):
    t0 = time.time()
    modelPrefix = 'models/RI/'
    resultPrefix = 'results/RI/'
    S = chooseData(data,nbr=simulatedSize)
    if data == "satellite":
        train = S
        test = chooseData("satellite test")
        if(addNoise):
            train = NoiseAdder(train)
            test = NoiseAdder(test)
    N = len(S)
    if(addNoise):
        S = NoiseAdder(S) # This function will add 10% noise on the data

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
    reportRI(selection, singleInput, onetree, data, S, runtime, latexTableFile)
    
if __name__=="__main__":
    try:
        os.remove(latexTableFile)
    except OSError:
        pass
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
