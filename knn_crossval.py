import pandas as pd
from ast import literal_eval
import KNN

dataset_dir = "./datasets/"

def preprocessTrainSet(trainSet):
	category_count = {c: 0 for c in set(trainSet["journeyPatternId"])}
	for jpid in trainSet["journeyPatternId"]:
		category_count[jpid] += 1
	newTrainSet_traj = []
	newTrainSet_jpid = []
	for row in trainSet.itertuples():
		if category_count[row[2]] >= 10:
			newTrainSet_traj.append(row[3])
			newTrainSet_jpid.append(row[2])
	return (newTrainSet_traj, newTrainSet_jpid)

# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
print "Loaded train_set."
trainSet = trainSet[0:500]

newTrainSet_traj, newTrainSet_jpid = preprocessTrainSet(trainSet)
print "Preprocessed data."
print len(newTrainSet_traj)

acc = KNN.crossvalidation(newTrainSet_traj, newTrainSet_jpid, 5)
print "Accuracy: " + str(acc)
