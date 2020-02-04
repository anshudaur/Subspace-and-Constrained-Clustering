from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics

pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def readSubSpaces():
   datawithclass=pd.read_csv('preprocessedData.csv')
   #f = open("testSubspaces_outputFile.txt", "r")
   #for row in f:
   filterData=datawithclass[subspaces]
   labelledDF = filterData.loc[((filterData['Class']== "A") | (filterData['Class']== "B") | (filterData['Class']== "C"))]
   return labelledDF

labelledDF=readSubSpaces()
Class=labelledDF['Class'];
Num=0;
catList= {};

for i in Class.unique():
    print(i)
    catList[i] = Num;
    Num += 1;
    
labelledDF['Class'].replace(catList, inplace=True)
ClassLabel=labelledDF['Class']

def normalise(df):
    print('normalizing')
	#df.fillna(df.mean(), inplace=True)
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_imputed_df = pd.DataFrame(x_scaled, columns = df.columns)
    return(X_imputed_df)


def objectToNumber(df):
    objectList = []
    categoryList = {};
    index = 0
    # objDF=pd.DataFrame();
    for cols in df.head():
        if(df[cols].dtype.name == 'object'):
            # print(df[cols] )
            objectList.append(index)
            df[cols] = df[cols].astype('category')
            categoryList[cols] = {};
            Num = 0;
            unique = df[cols].unique()
            for catVal in df[cols].unique():
                categoryList[cols][catVal] = Num;
                Num+=1;
            df[cols].replace(categoryList[cols], inplace = True)
            index+=1;
    df=normalise(df);
    return df;

labelledDF=objectToNumber(labelledDF.loc[:, labelledDF.columns != 'Class'])
labelledDF.reset_index(drop=True, inplace=True)
ClassLabel.reset_index(drop=True, inplace=True)
dataframe=pd.concat([labelledDF,ClassLabel],axis=1)
dataframe.fillna(labelledDF.mean(), inplace=True);

##Training via NB
gnb = GaussianNB()
# prepare cross validation
kfold = KFold(10, True, 1)
# enumerate splits
addAccuracy=0
addPrecision=0
addRecall=0
addF1score=0
addAUC=0
addKappa=0
i=0
for train, test in kfold.split(dataframe):
    # print(labelledDF.iloc[train])
    accuracy=0
    recall=0
    precision=0
    print('shapes of test and train',test.shape,train.shape)
    cvDataTrain=dataframe.iloc[train]
    classTrain=cvDataTrain['Class']
    cvDataTest=dataframe.iloc[test]
    classTest=cvDataTest['Class']
    # print(classTrain)
	# print('train: %s, test: %s' % (labelledDF[train], labelledDF[test]))
    gnb.fit(cvDataTrain.loc[:,cvDataTrain.columns != 'Class'],classTrain)
    predicted = gnb.predict(cvDataTest.loc[:,cvDataTest.columns != 'Class']);
    print("Accuracy:", metrics.accuracy_score(classTest, predicted))
    accuracy=metrics.accuracy_score(classTest, predicted)
    precision=metrics.precision_score(classTest,predicted,average='weighted')
    recall= metrics.recall_score(classTest,predicted,average='weighted')
    f1score=metrics.f1_score(classTest,predicted,average='weighted')

    # auc=metrics.roc_auc_score(classTest,predicted,average='weighted')
    kappa=metrics.cohen_kappa_score(classTest,predicted)
    print("kappa",kappa)
    addAccuracy=accuracy+addAccuracy
    addPrecision=precision+addPrecision
    addRecall=recall+addRecall
    addF1score=f1score+addF1score
    addKappa=kappa+addKappa    
    i=i+1


avgAccuracy=addAccuracy/i
avgPrecision=addPrecision/i
avgRecall=addRecall/i
avgF1score=addF1score/i
avgKappa=addKappa/i
# avgAUC=addAUC/i
print("i",i)
print("Average Accuracy",avgAccuracy)
print("Average Precion",avgPrecision)
print("Average Recall",avgRecall)
print("Average f1",avgF1score)
print("Average kappa",avgKappa)
