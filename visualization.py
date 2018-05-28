import gmplot
import pandas as pd
from ast import literal_eval
import os
import shutil

dataset_dir = "./datasets/"

# read trainSet
trainSet = pd.read_csv(
	dataset_dir + 'train_set.csv', 
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
print "Loaded trainSet"

vis_dir = "visualization"
if os.path.exists(vis_dir):
	shutil.rmtree(vis_dir, ignore_errors=True)
os.makedirs(vis_dir)

printed_ids = set()         # set of the ids of already printed trajectories
for row in trainSet.itertuples():
	if row[1] in printed_ids:
		continue
	else:
		printed_ids.add(row[1])
	longitudes = []
	latitudes = []
	longSum = 0
	latSum = 0
	for traj in row[2]:
		longitudes.append(traj[1])
		latitudes.append(traj[2])
		longSum += traj[1]
		latSum += traj[2]
	center = (longSum / len(row[2]), latSum / len(row[2]))
	gmap = gmplot.GoogleMapPlotter(center[1], center[0], 12)
	gmap.plot(latitudes, longitudes, 'green', edge_width=5)
	gmap.draw(vis_dir + "/map_{}.html".format(row[1]))
	print "Created map for JPID: " + str(row[1])
	if len(printed_ids) == 5:
		break
