import pandas as pd
import random
import sklearn as scikit
import hdbscan 
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import time
import sys
from generateConstraintsFile import *
from Evaluation import *

################ Class #######
class HEOM():
    def __init__(self, X, cat_ix, nan_equivalents = [np.nan, 0], normalised="normal"):
        """ Heterogeneous Euclidean-Overlap Metric
        Distance metric class which initializes the parameters
        used in heom function
        
        Parameters
        ----------
        X : array-like of shape = [n_rows, n_features]
            First instance 
        
        cat_ix : array-like of shape = [cat_columns_number]
            List containing categorical feature indices
        
        cat_ix : array-like of shape = [x]
            List containing missing values indicators
        normalised: string
            normalises euclidan distance function for numerical variables
            Can be set as "std". Default is a column range
        Returns
        -------
        None
        """      
        self.nan_eqvs = nan_equivalents
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        # Get the normalization scheme for numerical variables
        if normalised == "std":
            self.range = 4* np.nanstd(X, axis = 0)
        else:
            self.range = np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)
        
    def heom(self, x, y):
        """ Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        It can be used as a custom defined function for distance metrics
        in Scikit-Learn
        
        Parameters
        ----------
        x : array-like of shape = [n_features]
            First instance 
            
        y : array-like of shape = [n_features]
            Second instance
        Returns
        -------
        result: float
            Returns the result of the distance metrics function
        """
        # Initialise results' array
        results_array = np.zeros(x.shape)
        
        # Get indices for missing values, if any
        nan_x_ix = np.flatnonzero( np.logical_or(np.isin(x, self.nan_eqvs), np.isnan(x)))
        nan_y_ix = np.flatnonzero( np.logical_or(np.isin(y, self.nan_eqvs), np.isnan(y)))
        nan_ix = np.unique(np.concatenate((nan_x_ix, nan_y_ix)))
        # Calculate the distance for missing values elements
        results_array[nan_ix] = 1
        
        #print ("Type of cat_ix: ", type(self.cat_ix))
        if self.cat_ix:
            # Get categorical indices without missing values elements
            cat_ix = np.setdiff1d(self.cat_ix, nan_ix)
            # Calculate the distance for categorical elements
            results_array[cat_ix]= np.not_equal(x[cat_ix], y[cat_ix]) * 1 # use "* 1" to convert it into int
        
        # Get numerical indices without missing values elements
        num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        num_ix = np.setdiff1d(num_ix, nan_ix)
        # Calculate the distance for numerical elements
        results_array[num_ix] = np.abs(x[num_ix] - y[num_ix]) / self.range[num_ix]
        
        # Return the final result
        # Square root is not computed in practice
        # As it doesn't change similarity between instances
        return np.sum(np.square(results_array))
        
################ Class ENDS  #########################

#***************************** Quality Score Function **********************************
def readConstraints():
    mlc = {};  ### Must-Link-Constraints
    nlc = {};   ### Non-Link-Constraints
    ML = 0;    ## Count of ML
    NL = 0;        ## Total count of NL 
    #global ML, NL;
    ## Open the Must-Link constraint file in read mode
    f = open("mlc.txt", "r");
    for row in f:
        ## If the line is not empty, 
        if (len(row) > 0):
            ## The constraints consists of instance numbers. ( data points )
            row = row.split(',',1);            
            ML += 1;
            ## if either of the number is there in the dictionary already, just append the other.
            if int(row[0]) in mlc.keys():
                mlc[int(row[0])].append(int(row[1]));
            elif int(row[1]) in mlc.keys():
                mlc[int(row[1])].append(int(row[0]));
            ## If the combination does not exist, Add a new key.
            else:
                mlc[int(row[0])] = [int(row[1])];
    ## Open the Not-Link constraint file in read mode
    f = open("nlc.txt", "r");
    for row in f:
        ## If the line is not empty, 
        if (len(row) > 0):
            ## The constraints consists of instance numbers. ( data points )
            row = row.split(',',1);
            NL += 1;
            ## if either of the number is there in the dict already, just append the other.
            if int(row[0]) in nlc.keys():
                nlc[int(row[0])].append(int(row[1]));
            elif int(row[1]) in nlc.keys():
                nlc[int(row[1])].append(int(row[0]));
            ## If the combination does not exist, Add a new key.
            else:
                nlc[int(row[0])] = [int(row[1])];
                
    return(mlc,nlc, ML, NL);

## Function to calculate the quality of the subspace cluster.
def getQuality(currSubSpace, checkCluster, heomInst, mlc, nlc,  ML, NL):
    ## currSubSpace contains the instances for only the features in a given subSpace.
    ## CheckCluster has the list of all the instances as indexes 
    ##and its values are the clusters to which it belongs to.
    ## list = InstanceID[ ClusterID ]. It is used to check if the constraints are satisfied.
    ## heomInst contains the HEOM class instance for the given subSpace. 
    ## Parameters 1,2 and 4 are used to calculate the distance between NL and ML constraints
    
    mlDistSum = 0;     ## Sum of distances between ML constraints
    mlSat = 0;            ## Number of ML constraints satisfied
    nlDistSum =0;        ## Sum of distances between NL constraints
    nlSat = 0;            ## Number of NL constraints satisfied

    ## Check ML constraints by looping through ML constraints
    for i in mlc.keys(): 
        for j in mlc[i]:
            ## Calculate the distance between instance i and j and add it.
            mlDistSum = mlDistSum + heomInst.heom(currSubSpace.values[i], currSubSpace.values[j]);
            ## Check if the instances belong to the same cluster or not.
            if checkCluster[i] == checkCluster[j] :
                mlSat += 1;
    
    ## Check NL constraints by looping through NL constraints
    for i in nlc.keys(): 
        for j in nlc[i]:
            ## Calculate the distance between j and k and add it.
            nlDistSum = nlDistSum + heomInst.heom(currSubSpace.values[i], currSubSpace.values[j]); 
            ## Check if the instances belong to the same cluster or not.
            if checkCluster[i] != checkCluster[j] :
                nlSat += 1;
                
    if (ML+NL) > 0:
        qcons = (mlSat + nlSat) / (ML + NL);
    else: 
        qcons = 0;
    if NL > 0 and ML > 0:
        qdist = (nlDistSum/NL) - (mlDistSum/ML);
    elif ML > 0: 
        qdist = (mlDistSum/ML);
    elif NL > 0:
        qdist = (nlDistSum/NL)
    else: 
        qdist = 0;
    quality = qcons*qdist;
    print("Quality is :", quality)
    return (quality);
    
## Function to calculate the quality of the subspace cluster.
def getQdist(currSubSpace, heomInst, mlc, nlc, ML, NL):
    ## currSubSpace contains the instances for only the features in a given subSpace.
    ## CheckCluster has the list of all the instances as indexes 
    ##and its values are the clusters to which it belongs to.
    ## list = InstanceID[ ClusterID ]. It is used to check if the constraints are satisfied.
    ## heomInst contains the HEOM class instance for the given subSpace. 
    ## Parameters 1,2 and 4 are used to calculate the distance between NL and ML constraints
    
    mlDistSum = 0;     ## Sum of distances between ML constraints
    nlDistSum =0;        ## Sum of distances between NL constraints

    ## Check ML constraints by looping through ML constraints
    for i in mlc.keys(): 
        for j in mlc[i]:
            ## Calculate the distance between instance i and j and add it.
            mlDistSum = mlDistSum + heomInst.heom(currSubSpace.values[i], currSubSpace.values[j]);
            
    ## Check NL constraints by looping through NL constraints
    for i in nlc.keys(): 
        for j in nlc[i]:
            ## Calculate the distance between j and k and add it.
            nlDistSum = nlDistSum + heomInst.heom(currSubSpace.values[i], currSubSpace.values[j]);
    
    if NL > 0 and ML > 0:
        qdist = (nlDistSum/NL) - (mlDistSum/ML);
    elif ML > 0: 
        qdist = (mlDistSum/ML);
    elif NL > 0:
        qdist = (nlDistSum/NL)
    else: 
        qdist = 0;
    print("Quality qdist is :", qdist)
    return (qdist);
#***************************** Quality Score Function ENDS *****************************

#*****************************Main Algorithm********************************************
def normalise(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_imputed_df = pd.DataFrame(x_scaled, columns = df.columns)
    return(X_imputed_df)

def coreFunction(df, subSpace, subSpaceCatList, algo2, sDistCan, categoryList, mlc, nlc,  ML, NL):
    column = df.columns[0]
    filterCheckPassed = 1;
    Clusters=list();
    subSpaceSet = {};
    subSpaceStarList = [];
    sDist = 0;
    keyList = [];
    keyList = subSpace.columns.values.tolist();
    
    ## Since we have changed the data type of the originally read data, 
    ## Compare the column with the categorical keys to find the categorical values.
    ## If the index, matches, append subSpaceFeatIndex, because 
    ## in the subSpace DataFrame, the index of the categorical value is subSpaceFeatIndex
    ## Add the column to the temp DataFrame variable subSpace    
    if column in categoryList.keys():
        subSpaceCatList.append(len(keyList));
    keyList.append(column);
    print("Clustering for - ", keyList)
    subSpace[column] = df[column] 
    heom_metric1 = HEOM(subSpace, subSpaceCatList, normalised="std")
    
    ## For Algorithm 2, filter criteria, get the sDist
    if algo2 == 1:
        if column in categoryList.keys():
            subSpaceStarList.append(0);
        ## Get the distance of the S* feature. 
        heom_metricStar = HEOM(df, subSpaceStarList, normalised="std")
        sDistStar = getQdist(df, heom_metricStar, mlc, nlc,  ML, NL)
        subSpaceStarList.clear()
        ## Compare S* distance and Candidate subSpace distance.
        ## Whichever is greater, assign it to the sDist
        if  sDistStar > sDistCan:
            sDist = sDistStar
        else:
            sDist = sDistCan;
        ### Compare the distance of the new subSpace ( after merging ) 
        ### With that of the sDist. (Filter criteria) . IFF its greater, form clusters.
        newSubSpaceDist = getQdist(subSpace, heom_metric1, mlc, nlc,  ML, NL)
        ## If the condition fails, make sure, the cluster is not evaluated.
        if newSubSpaceDist < sDist:
            filterCheckPassed = 0;
            print("Filter not passed for the subSpace:", keyList);
            
    if filterCheckPassed == 1 :
        #HDBSCAN
        coredist_jobs = mp.cpu_count();
        clusterer1 = hdbscan.HDBSCAN(min_cluster_size=8, metric=heom_metric1.heom, core_dist_n_jobs = coredist_jobs)
        
        tempSubSpace = subSpace;
        tempSubSpace.fillna(tempSubSpace.mean(), inplace=True) 
        cluster_labels = clusterer1.fit(tempSubSpace)
        
        #It is an array with an integer for each data sample. Samples
        #that are in the same cluster get assigned the same number`
        subSpaceCluster = clusterer1.labels_.tolist();
        Clusters.append(clusterer1.labels_.tolist());
        
        quality = getQuality(subSpace, subSpaceCluster, heom_metric1, mlc, nlc,  ML, NL)
        ### Store the subSpaces and its clusters and quality in a dictionary.
        ### name of the Features in the subSpace is the key. Value is the dictionary which has cluster and quality as the key value pair. 
        subSpaceKey = ",".join(keyList)### Using this, we can store the subSpaces with more than 1 feature as a key in dictionary.
        subSpaceSet[subSpaceKey] = {} ### When it is more than 1 feature in a subSpace, use the subSpaceKey as the key to the dictionary
        subSpaceSet[subSpaceKey]["cluster"] = subSpaceCluster
        subSpaceSet[subSpaceKey]["quality"] = quality
    
    subSpace.drop(column, axis=1, inplace=True);
    keyList.remove(column);
    if column in categoryList.keys():
        subSpaceCatList.remove(len(keyList)) 
        
    return subSpaceSet; 
    
def getSubSpaceClusters(df, subSpace, subSpaceCatList, categoryList, mlc, nlc,  ML, NL):
    algo2 = 0;            ## Variable used to check the filter condition.   1- check filter condition 0 - do not check.
    sDistCan  = 0;
    subSpaceSet = {}
    subSpaceSet["qBest"]  = 0;
    subSpaceSet["qBestSubSpace"] =  '';
    
    ## If the subSpace is not empty, it is called by the algorithm 2.
    ## So, it is the candidateSubSpace and find the distance of the candidate subspace.
    ## And mark the algo2 to be 1.
    if subSpace.empty is False:
        algo2 = 1;
        heom_metric = HEOM(subSpace, subSpaceCatList, normalised="std")
        sDistCan = getQdist(subSpace, heom_metric, mlc, nlc,  ML, NL)
    
    ## Use parallelization.
    pool = mp.Pool(mp.cpu_count());
    coreFunctionPartial = partial(coreFunction, subSpace=subSpace, subSpaceCatList=subSpaceCatList, algo2=algo2, sDistCan = sDistCan ,categoryList = categoryList, mlc = mlc, nlc=nlc,  ML=ML, NL=NL)
    result = pool.map(coreFunctionPartial,  ([df.loc[:,[cols]] for cols in df]));
    pool.close();
    pool.join()
    
    sys.stdout.flush();
    
    ## Aggregate the result obtained from different threads.
    for i in result:
        for key in i.keys():
            subSpaceSet[key] = i[key];
            if i[key]["quality"] >= subSpaceSet["qBest"]:
                subSpaceSet["qBest"] = i[key]["quality"]
                subSpaceSet["qBestSubSpace"] = key;
    ## Return the aggregated value
    return subSpaceSet;
    
def dressPlus(mlc, nlc, ML, NL):
    df=pd.read_csv('preprocessedData.csv')
    if 'Class' in df.columns:
        df = df.drop(['Class'],axis=1)
    if 'Index' in df.columns:
        df = df.drop(['Index'],axis=1)
    
    objectList=[]
    canSubSpaceObjList = [];
    categoryList = {};
    candidateSubSpaceSet = {};
    candidateSubSpace = pd.DataFrame()
    qBest = 0;
    condition = True;
    subSpaceCluster = {};
    subSpaceCluster["qBest"] = 0;
    subSpaceCluster["qBestSubSpace"] = '';
    
    ### Read the dataset into the DataFrame and normalize
    index=0
    for cols in df.head():
        temp = 1;    
        if(df[cols].dtype.name == 'object'):
            objectList.append(index)
            df[cols] = df[cols].astype('category')
            categoryList[cols] = {};
            Num = 0;
            unique = df[cols].unique() 
            unique = unique.dropna()
            for catVal in unique:
                categoryList[cols][catVal] = Num;
                Num+=1;
            df[cols].replace(categoryList[cols], inplace = True)
        index+=1
    df=normalise(df);
    
    ## The function returns the dictionary having the subSpaces are keys.
    ## Values for the subspaces are the clusters, quality.
    ## The dictionary also maintains the qBest value and its corresponding subSpace.
    subSpaceCluster = getSubSpaceClusters(df, candidateSubSpace, canSubSpaceObjList, categoryList, mlc, nlc,  ML, NL);
    qBest = subSpaceCluster["qBest"];
    print("The best subSpace and quality is: ", subSpaceCluster["qBestSubSpace"], subSpaceCluster["qBest"] );
    ## ********* Algorithm 1 ends here. **********
    
    ## *************** HERE COMES OUTSIDE WHILE LOOP of algo2.************* ##
    ## The loop runs till the feature set is exhausted 
    ## OR the new qBest (candidate qBest) is not better than the existing qBest.
    if qBest <= 0:
        condition = False;
    while condition is True:
        ## Now, we have the qBest feature. This will be added to the candidate subSpace and removed from the S.
        ## S is the subset of feature set. Each feature in S is S*
        if not subSpaceCluster["qBestSubSpace"]:
            condition = False;
            break;
        
        candidateSubSpaceSet[subSpaceCluster["qBestSubSpace"]] =  subSpaceCluster[subSpaceCluster["qBestSubSpace"]] ## This is stored just for reference. to maintain the record. 
        
        index = 0;
        for i in subSpaceCluster["qBestSubSpace"].split(",") :
            if i in df.columns:
                candidateSubSpace[i] = df[i];       ## Copy the chosen features to candidateSubSpace. Which will be merged with others.
                df.drop(i, axis=1, inplace=True);
                ## For the candidateSubSpace, prepare the objectList.
                if i in categoryList.keys():
                    if canSubSpaceObjList:
                        canSubSpaceObjList.append(index);
            index+=1;
            
        
        if df.empty is True: ## Break the loop if all the features in the set S is checked. 
                break;
        ### Get the clusters for the new subspace. merging with candidateSubSpace
        ### Filter condition is checked in the function "getSubSpaceClusters"
        print("Clustering further with the new subspace:", candidateSubSpace.columns);
        subSpaceCluster = getSubSpaceClusters(df, candidateSubSpace, canSubSpaceObjList, categoryList, mlc, nlc, ML, NL);
        ## If the cluster quality is better than the existing, 
        ## Only then, merge this with the candidateSubSpace.
        print("qBest: ", qBest, "new Best: ", subSpaceCluster["qBest"]);
        if (subSpaceCluster["qBest"] <= qBest) or (subSpaceCluster["qBest"] <= 0.0) : ## possible its not there.
            condition = False; ## Break the condition because the candidate set is empty.
        elif (subSpaceCluster["qBest"] > qBest):
            qBest = subSpaceCluster["qBest"] ;
    f = open("output.txt","w")        
    print("Final subSpace cluster is: ", subSpaceCluster["qBestSubSpace"], " quality is:", subSpaceCluster["qBest"] );
    print("Overall Chosen Clusters are -")
    
    for i in candidateSubSpaceSet.keys():
        #f.write("%s\r\n, %s\r " %(i, candidateSubSpaceSet[i]["quality"] ))
        f.write(str(i))
        print("SubSpace: ", i, "Quality: ", candidateSubSpaceSet[i]["quality"] );
#*****************************Main Algorithm ENDS ********************************************

def main():
    print("Dress+ execution start , Parallelization is done with", mp.cpu_count(), "Number of processors")
    mlc = {};
    nlc = {}
    generateConstraints();
    start_time = time.time()
    dressPlus(mlc, nlc, ML, NL);
    end_time = time.time()
    print("Total execution time: {}".format(end_time - start_time))
    
if __name__ == '__main__':
    mp.freeze_support()
    main();