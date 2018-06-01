import gmplot
import pandas as pd
import numpy as np
from ast import literal_eval
from fastdtw import fastdtw
import util
import time
import os
import shutil

dataset_dir = "./datasets/"

# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
with open(dataset_dir + 'test_set_a1.csv') as f:
	next(f)		# skip first line
	test_trajectories = [literal_eval(line.rstrip("\n")) for line in f]
print "Loaded datasets."

dtw_dir = "dtw_results"
if os.path.exists(dtw_dir):
	shutil.rmtree(dtw_dir, ignore_errors=True)

for tid, traj in enumerate(test_trajectories):
	min5 = [float('inf')]*5
	patternIds = [None]*5
	paths = [None]*5
	start_time = time.time()
	x = np.array([[z[1], z[2]] for z in traj])
	for row in trainSet.itertuples():
		y = np.array([[z[1], z[2]] for z in row[2]])
		distance, _ = fastdtw(x, y, dist=util.harversineDist)
		curMax, maxpos = max((v, i) for i, v in enumerate(min5))
		if distance < curMax:        # new distance is smaller that min5's largest distance
			min5[maxpos] = distance  # so replace it with it
			paths[maxpos] = y
			patternIds[maxpos] = row[1]
	Dt = int(time.time() - start_time)
	print "Calculated 5 nearest neighbours for trajectory {}.".format(tid)
	print "Dt = " + str(Dt) + "sec"
	print "5 nearest neighbours are: "
	min5, paths, patternIds = zip(*sorted(zip(min5, paths, patternIds)))		# sorts lists based on min5
	print min5
	min5 = [round(i, 1) for i in min5]
	print patternIds
	print '\n'
	# plotting maps:
	traj_dir = dtw_dir + "/trajectory{}".format(tid)
	os.makedirs(traj_dir)
	util.plotMap(x, traj_dir + "/Test_traj_{}sec.html".format(Dt))
	for i, y in enumerate(paths):
		util.plotMap(y, traj_dir + "/N{}_{}_{}.html".format(i + 1, patternIds[i], min5[i]))


