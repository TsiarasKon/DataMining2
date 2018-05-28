import pandas as pd
from ast import literal_eval
import KNN

dataset_dir = "./datasets/"


# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
with open(dataset_dir + 'test_set_a2.csv') as f:
	next(f)		# skip first line
	test_trajectories = [literal_eval(line.rstrip("\n")) for line in f]
print "Loaded datasets."

clf = KNN.KNN(5)
clf.fit(trainSet["Trajectory"], trainSet["journeyPatternId"])
predicted_JPIDS = clf.predict(test_trajectories)

cvFile = open("./testSet_JourneyPatternIDs.csv", "w+")

cvFile.write("Test_Trip_ID\tPredicted_JourneyPatternID\n")
for i in range(len(predicted_JPIDS)):
	cvFile.write(str(i) + '\t' + predicted_JPIDS[i] + '\n')

cvFile.close()
