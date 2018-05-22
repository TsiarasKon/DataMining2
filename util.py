import math

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
