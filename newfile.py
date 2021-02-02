# Libraries
import os
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

PATH="HMP_dataset/"
#Constants
NUM_CLUSTERS_L1 = 40
NUM_CLUSTERS_L2 = 12


# Generate training and testing data
def readData():
    activityNames = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
    dataset = []
    labels = []
    for i in range(len(activityNames)):
        for fileName in os.listdir('HMP_Dataset/' + activityNames[i]):
            data = pd.read_csv('HMP_Dataset/' + activityNames[i] + '/' + fileName, sep=" ", header=None)
            dataset.append(data.values.flatten())
            labels.append(i)
    return dataset, labels



def subSignal(dataSet,SUBSET_LEN):
  subsets = []
  subsetVector = []
  for i in range(len(dataSet)):
      subsetVector.append([])
      for j in range(int(len(dataSet[i]) / SUBSET_LEN)):
          subsetRangeA = (j * SUBSET_LEN)
          subsetRangeB = (j * SUBSET_LEN + SUBSET_LEN)
          subsets.append(dataSet[i][subsetRangeA : subsetRangeB])
          subsetVector[i].append(dataSet[i][subsetRangeA : subsetRangeB])
  return subsets, subsetVector



def train(subsets):
    kMeansL1 = KMeans(n_clusters = NUM_CLUSTERS_L1)
    kMeansL1.fit(subsets)
    kMeansL2=[]
    dataCluster = []
    for i in range(NUM_CLUSTERS_L1):
        dataCluster.append([])
    for idx, elem in enumerate(subsets):
        index = kMeansL1.predict([elem])[0];
        dataCluster[index].append(elem);
    cluster2Count = 0
    for i in range(NUM_CLUSTERS_L1):
        cluster2Count = NUM_CLUSTERS_L2
        if len(dataCluster[i]) < NUM_CLUSTERS_L2:
            cluster2Count = len(dataCluster[i])
        kMeansL2.append(KMeans(n_clusters = cluster2Count))
        kMeansL2[i].fit(dataCluster[i])
    return kMeansL1, kMeansL2, cluster2Count


def generateFeatures(subsetVector, kMeansL1, kMeansL2, xxxx):
    numClusters = xxxx
    featureVector = np.zeros((len(subsetVector), numClusters))
    for i, elem in enumerate(subsetVector):
        for j in elem:
            predictionL1 = kMeansL1.predict([j])[0]
            predictionL2 = kMeansL2[predictionL1].predict([j])[0]
            indexCluster = predictionL1 * NUM_CLUSTERS_L2 + predictionL2
            featureVector[i][indexCluster] += 1

    featureVector = np.array(featureVector)

    return featureVector


def booter(input):
    dataset, labels = readData()
    trainData, testData, trainLabel, testLabel = train_test_split(dataset, labels, test_size=0.2, train_size=0.8)
    subsets, subsetVector = subSignal(trainData, input)
    kMeansL1, kMeansL2, xxxx = train(subsets)
    featureVector = generateFeatures(subsetVector, kMeansL1, kMeansL2, NUM_CLUSTERS_L1*xxxx)
    randomForest = RandomForestClassifier(n_jobs = -1)
    randomForest.fit(featureVector, trainLabel)
    subsetsTest, subsetsVectorTest = subSignal(testData, input)
    featureVector2 = generateFeatures(subsetsVectorTest, kMeansL1, kMeansL2, NUM_CLUSTERS_L1*xxxx)
    predictionAccuracy = randomForest.score(featureVector2, testLabel)
    predictions = []
    for feature in (featureVector2):
        predictions.append(randomForest.predict([feature]))
    # print( confusion_matrix(y_test, pred))
    return predictionAccuracy

def main():
    maxval=0
    maxindex=0
    constantavg=5 #averaging results
    for i in range(1, 100):  #range of subsignal lengths
        acc=0
        for j in range(constantavg):
            acc += booter(i)
        if acc>0:
            acc= acc/constantavg
        if(acc>maxval):
            maxval=acc
            maxindex=i
    return maxindex, maxval

print(main())
