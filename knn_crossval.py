import pandas as pd
from ast import literal_eval
import KNN

dataset_dir = "./datasets/"


# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
print "Loaded train_set."
trainSet = trainSet[0:500]

acc = KNN.crossvalidation(trainSet["Trajectory"], trainSet["journeyPatternId"], 5)
print "Accuracy: " + str(acc)
