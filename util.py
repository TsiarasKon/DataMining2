import math
import gmplot

def harversineDist(p1, p2):
	l1 = math.radians(p1[0])
	l2 = math.radians(p2[0])
	f1 = math.radians(p1[1])
	f2 = math.radians(p2[1])
	dl = l2 - l1		# longitude difference
	df = f2 - f1		# latitude difference
	a = (math.sin(df / 2.0) ** 2) + math.cos(f1) * math.cos(f2) * (math.sin(dl / 2.0) ** 2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	R = 6371
	return R * c


def matching_function(x, y, threshold=0.2, dist_funct=harversineDist):
	return (dist_funct(x, y) <= threshold)


def plotMap(coordList, name, subtrajectory=None):
	longitudes = []
	latitudes = []
	longSum = 0
	latSum = 0
	for coord in coordList:
		longitudes.append(coord[0])
		latitudes.append(coord[1])
		longSum += coord[0]
		latSum += coord[1]
	center = (longSum / len(coordList), latSum / len(coordList))
	gmap = gmplot.GoogleMapPlotter(center[1], center[0], 12)
	gmap.plot(latitudes, longitudes, 'green', edge_width=5)
	if subtrajectory is not None:
		for i in range(len(coordList)):
			if matching_function(coordList[i], subtrajectory[0]):
				break
		j = 0
		while j < len(subtrajectory):
			longitudes = []
			latitudes = []
			while matching_function(coordList[i], subtrajectory[j]):
				longitudes.append(subtrajectory[j][0])
				latitudes.append(subtrajectory[j][1])
				i += 1
				j += 1
			if longitudes != []:
				gmap.plot(latitudes, longitudes, 'red', edge_width=5)
			j += 1
	gmap.draw(name)
