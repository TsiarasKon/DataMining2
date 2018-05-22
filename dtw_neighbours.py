import gmplot
import pandas as pd
import numpy as np
from ast import literal_eval
from io import BytesIO
from PIL import Image
import urllib
from fastdtw import fastdtw
from util import harversineDist

# read sets
trainSet = pd.read_csv(
	'train_set.csv', 
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
with open('test_set_a1.csv') as f:
	next(f)		# skip first line
	test_trajectories = [literal_eval(line.rstrip("\n")) for line in f]
print "Loaded datasets."

min5 = [float('inf')]*5
patternIds = [None]*5
paths = [None]*5
for traj in test_trajectories[0]:
	x = np.array([[z[1], z[2]] for z in traj])
	for row in trainSet.itertuples():
		y = np.array([[z[1], z[2]] for z in row[2]])
		distance, path = fastdtw(x, y, dist=harversineDist)
		curMax, maxpos = max((v, i) for i, v in enumerate(min5))
		if distance < curMax:        # new distance is smaller that min5's largest distance
			min5[maxpos] = distance  # so replace it with it
			paths[maxpos] = path
			patternIds[maxpos] = row[1]
	print "Calculated traj"
	print "5 nearest neighbours are: "
	print min5
	print patternIds



