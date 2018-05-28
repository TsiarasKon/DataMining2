import pandas as pd
from ast import literal_eval
import KNN

dataset_dir = "./datasets/"


# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
trainSet_traj = []
trainSet_jpid = []
for row in trainSet.itertuples():
	trainSet_traj.append([[z[1], z[2]] for z in row[3]])
	trainSet_jpid.append(row[2])

with open(dataset_dir + 'test_set_a2.csv') as f:
	next(f)		# skip first line
	testSet = [literal_eval(line.rstrip("\n")) for line in f]
testSet_traj = []
for traj in testSet:
	testSet_traj.append([[z[1], z[2]] for z in traj])
print "Loaded datasets."

clf = KNN.KNN(5)
clf.fit(trainSet_traj, trainSet_jpid)
predicted_JPIDS = clf.predict(testSet_traj)

cvFile = open("./testSet_JourneyPatternIDs.csv", "w+")

cvFile.write("Test_Trip_ID\tPredicted_JourneyPatternID\n")
for i in range(len(predicted_JPIDS)):
	cvFile.write(str(i) + '\t' + predicted_JPIDS[i] + '\n')

cvFile.close()
