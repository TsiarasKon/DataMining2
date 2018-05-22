import gmplot
import pandas as pd
from ast import literal_eval
from io import BytesIO
from PIL import Image
import urllib

# read trainSet
trainSet = pd.read_csv(
	'train_set.csv', 
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
print "Loaded trainSet"

printed_ids = set()         # set of the ids of already printed trajectories
for row in trainSet.itertuples():
	if row[0] in printed_ids:
		continue
	else:
		printed_ids.add(row[0])
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
	url = "mymap{}.html".format(row[0])
	gmap.draw(url)
	buff = BytesIO(urllib.urlopen(url).read())
	image = Image.open(buff)
	image.save("map{}.png".format(row[0]))
	print "Created image for Tripid:" + str(row[0])
	if len(printed_ids) == 5:
		break
