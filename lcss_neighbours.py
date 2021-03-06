import gmplot
import pandas as pd
import numpy as np
from ast import literal_eval
import util
import time
import os
import shutil
import LCSS

dataset_dir = "./datasets/"

# read sets
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
with open(dataset_dir + 'test_set_a2.csv') as f:
	next(f)		# skip first line
	test_trajectories = [literal_eval(line.rstrip("\n")) for line in f]
print "Loaded datasets."

lcss_dir = "lcss_results"
if os.path.exists(lcss_dir):
	shutil.rmtree(lcss_dir, ignore_errors=True)

for tid, traj in enumerate(test_trajectories):
	max5 = [-1]*5
	patternIds = [None]*5
	paths = [None]*5
	common_subtrajectories = [None]*5
	start_time = time.time()
	x = np.array([[z[1], z[2]] for z in traj])
	for row in trainSet.itertuples():
		y = np.array([[z[1], z[2]] for z in row[2]])
		C = LCSS.LCSS(x, y, util.matching_function)
		lenx = len(x)
		leny = len(y)
		score = C[lenx][leny]
		curMin, minpos = min((v, i) for i, v in enumerate(max5))
		if score > curMin:        	# new score is greater than max5 lowest score
			max5[minpos] = score  	# so replace it with it
			paths[minpos] = y
			patternIds[minpos] = row[1]
			common_subtrajectories[minpos] = LCSS.findSolution(C, x, y, lenx, leny, util.matching_function)
	Dt = int(time.time() - start_time)
	print "Calculated 5 nearest neighbours for trajectory {}.".format(tid)
	print "Dt = " + str(Dt) + "sec"
	print "5 nearest neighbours are: "
	#max5, paths, patternIds, common_subtrajectories = zip(*sorted(zip(max5, paths, patternIds, common_subtrajectories)))		# sorts lists based on min5
	print max5
	print patternIds
	print '\n'
	# plotting maps:
	traj_dir = lcss_dir + "/trajectory{}".format(tid)
	os.makedirs(traj_dir)
	util.plotMap(x, traj_dir + "/Test_traj_{}sec.html".format(Dt))
	for i, y in enumerate(paths):
		util.plotMap(y, traj_dir + "/N{}_{}_{}.html".format(i + 1, patternIds[i], max5[i]), common_subtrajectories[i])
