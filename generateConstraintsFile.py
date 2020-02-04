import errno
import os
import pandas as pd
from collections import defaultdict
from itertools import combinations
from itertools import product
from itertools import chain
from pprint import pprint
from array import *

def generateConstraints():
    fileName = r'E:\KMDProject\medical-mining\Dress+ Final Code\preprocessedData.csv'
    rowDict = {}
    colLabelDict = defaultdict(dict)    
    MLConstraints = defaultdict(dict)
    NLCombinationLists = []
    MLCombinationLists = []
    NLConstraints =  defaultdict(dict)
    uniqueClasses =[]
    finalMLList=""
    #finalNLList = []
    #finalMLList = []
    #qCons, qDist ,qBest,qScore = 0
    if os.stat(fileName).st_size > 0:
        try:
            dataDf = pd.read_csv(fileName, header = 0)
            dataDf.set_index('Index')
            uniqueClasses = pd.Series(dataDf['Class'].unique())
            uniqueClasses = uniqueClasses.dropna()
            splitClassData = dataDf.groupby('Class').groups
                      
            for key in splitClassData:
                #print(key , splitClassData[key].values)
                #NLList.append(splitClassData[key].values.tolist())
                NLConstraints[key] = splitClassData[key].values.tolist()
                MLConstraints[key] = set(list(combinations(splitClassData[key].values, 2)))  #splitClassData[key].values
            
            keyPairwiseArr = list(combinations(splitClassData.keys(), 2))
            for elem in keyPairwiseArr:
                pairkey1 = elem[0]
                pairkey2 = elem[1]                
                NLCombinationLists = list(product(splitClassData[pairkey1].values.tolist(), splitClassData[pairkey2].values.tolist()))
            
            mlcFile = open('mlc.txt', 'w')
            for key,val in MLConstraints.items():
                for item1,item2 in val:
                    mlcFile.write(str(item1)+","+str(item2)+"\n")
            mlcFile.close()
            nlcFile = open('nlc.txt', 'w')
            for val in NLCombinationLists:
                nlcFile.write(str(val[0])+","+str(val[1])+"\n")
            nlcFile.close()
            
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise