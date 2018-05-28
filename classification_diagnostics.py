import pandas as pd
from ast import literal_eval
import KNN

dataset_dir = "./datasets/"


# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval}
)
category_count = {c: 0 for c in set(trainSet["journeyPatternId"])}
for jpid in trainSet["journeyPatternId"]:
	category_count[jpid] += 1

print category_count
average = sum(category_count.itervalues()) / float(len(category_count))
print "Average: " + str(average)
