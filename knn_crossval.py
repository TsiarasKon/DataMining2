import pandas as pd
from ast import literal_eval
import KNN

# read sets
trainSet = pd.read_csv(
	'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
print "Loaded train_set."
trainSet = trainSet[0:1000]

acc = KNN.crossvalidation(trainSet["Trajectory"], trainSet["journeyPatternId"], 5)
print "Accuracy: " + str(acc)
