import gmplot
import pandas as pd
from ast import literal_eval
from io import BytesIO
from PIL import Image
import urllib

# hardcoded trajectory for debugging:
# trajectory = [[1353943168000000.0, -6.237581, 53.383015], [1353943186000000.0, -6.2388959999999996, 53.38140500000001], [1353943205000000.0, -6.2394169999999995, 53.381092], [1353943226000000.0, -6.241379, 53.381565], [1353943246000000.0, -6.242865, 53.381919999999994], [1353943267000000.0, -6.243486, 53.382103], [1353943287000000.0, -6.243485, 53.382107], [1353943306000000.0, -6.243485, 53.382107], [1353943326000000.0, -6.243485, 53.382107], [1353943345000000.0, -6.243485, 53.382107], [1353943365000000.0, -6.244606, 53.382256000000005], [1353943388000000.0, -6.244606, 53.382256000000005], [1353943406000000.0, -6.244606, 53.382256000000005], [1353943427000000.0, -6.245975, 53.380276], [1353943445000000.0, -6.246649, 53.379311], [1353943466000000.0, -6.2477480000000005, 53.37809], [1353943488000000.0, -6.248648, 53.377143999999994], [1353943504000000.0, -6.250482, 53.375668000000005], [1353943527000000.0, -6.250914, 53.37516], [1353943545000000.0, -6.251738, 53.373955], [1353943566000000.0, -6.252235, 53.373093000000004], [1353943588000000.0, -6.252987999999999, 53.371937], [1353943605000000.0, -6.2539, 53.370518000000004], [1353943627000000.0, -6.254246, 53.369949], [1353943646000000.0, -6.255113, 53.368481], [1353943666000000.0, -6.255481, 53.36715699999999], [1353943687000000.0, -6.255481, 53.36715699999999], [1353943705000000.0, -6.255771, 53.366165], [1353943726000000.0, -6.257155, 53.364243], [1353943744000000.0, -6.257567, 53.363785], [1353943767000000.0, -6.258814, 53.362438], [1353943785000000.0, -6.2594330000000005, 53.361759], [1353943806000000.0, -6.260783, 53.360271], [1353943832000000.0, -6.26098, 53.360065000000006], [1353943866000000.0, -6.262354, 53.35866899999999], [1353943886000000.0, -6.263205, 53.358002], [1353943905000000.0, -6.263332, 53.357898999999996], [1353943925000000.0, -6.263332, 53.357898999999996], [1353943946000000.0, -6.263575, 53.3577], [1353943968000000.0, -6.264519, 53.3568], [1353943985000000.0, -6.264897, 53.356415000000005], [1353944005000000.0, -6.268148, 53.353106999999994], [1353944026000000.0, -6.265179, 53.352104000000004], [1353944046000000.0, -6.26514, 53.352081000000005], [1353944069000000.0, -6.263403, 53.354206000000005], [1353944085000000.0, -6.262194, 53.353352], [1353944106000000.0, -6.261166, 53.352329000000005], [1353944128000000.0, -6.261029, 53.352008999999995], [1353944145000000.0, -6.2607870000000005, 53.351448], [1353944167000000.0, -6.260475, 53.350731], [1353944186000000.0, -6.260426, 53.35061999999999], [1353944204000000.0, -6.260426, 53.35061999999999], [1353944225000000.0, -6.260426, 53.35061999999999], [1353944245000000.0, -6.260426, 53.35061999999999], [1353944266000000.0, -6.260317, 53.350372], [1353944284000000.0, -6.260066999999999, 53.349833999999994], [1353944307000000.0, -6.259402, 53.348282], [1353944327000000.0, -6.259318, 53.348068000000005], [1353944345000000.0, -6.259318, 53.348068000000005], [1353944366000000.0, -6.259318, 53.348068000000005], [1353944388000000.0, -6.259318, 53.348068000000005], [1353944405000000.0, -6.258793, 53.347069], [1353944425000000.0, -6.258214, 53.346489], [1353944468000000.0, -6.257535, 53.346012], [1353944490000000.0, -6.257535, 53.346012], [1353944509000000.0, -6.257349, 53.345776], [1353944527000000.0, -6.25905, 53.344978000000005], [1353944548000000.0, -6.260922, 53.344280000000005], [1353944569000000.0, -6.26169, 53.344311], [1353944588000000.0, -6.26169, 53.344311], [1353944608000000.0, -6.262094, 53.344292], [1353944630000000.0, -6.26377, 53.343979000000004], [1353944648000000.0, -6.264412999999999, 53.343403], [1353944669000000.0, -6.264585, 53.3424], [1353944688000000.0, -6.264585, 53.3424], [1353944708000000.0, -6.264887, 53.341984], [1353944729000000.0, -6.265084, 53.341784999999994], [1353944747000000.0, -6.265084, 53.341784999999994], [1353944768000000.0, -6.265568, 53.340866000000005], [1353944790000000.0, -6.26579, 53.33993100000001], [1353944808000000.0, -6.26579, 53.33993100000001], [1353944828000000.0, -6.265841, 53.33969499999999], [1353944851000000.0, -6.266058, 53.338249], [1353944869000000.0, -6.2659910000000005, 53.338078], [1353944889000000.0, -6.2659910000000005, 53.338078], [1353944908000000.0, -6.266001999999999, 53.338104], [1353944928000000.0, -6.266001999999999, 53.338104], [1353944950000000.0, -6.265738, 53.337322], [1353944967000000.0, -6.265379, 53.336040000000004], [1353944988000000.0, -6.265255000000001, 53.335097999999995], [1353945008000000.0, -6.265276, 53.334385], [1353945027000000.0, -6.265178, 53.333828000000004], [1353945047000000.0, -6.265178, 53.333828000000004], [1353945070000000.0, -6.265178, 53.333828000000004], [1353945087000000.0, -6.264107, 53.333691], [1353945089000000.0, -6.264107, 53.333691], [1353945107000000.0, -6.262643, 53.33301899999999], [1353945130000000.0, -6.262819, 53.332767000000004], [1353945149000000.0, -6.264598, 53.332606999999996], [1353945167000000.0, -6.266115, 53.332554], [1353945190000000.0, -6.268909, 53.332432], [1353945209000000.0, -6.27186, 53.332283], [1353945229000000.0, -6.2740279999999995, 53.332191], [1353945248000000.0, -6.274299, 53.332203], [1353945268000000.0, -6.274299, 53.332203], [1353945289000000.0, -6.2745239999999995, 53.332211], [1353945308000000.0, -6.274780000000001, 53.332221999999994], [1353945328000000.0, -6.275228, 53.331154000000005], [1353945349000000.0, -6.2753, 53.330563], [1353945367000000.0, -6.275965, 53.328991], [1353945388000000.0, -6.27637, 53.328506000000004], [1353945411000000.0, -6.27637, 53.328506000000004], [1353945427000000.0, -6.277548, 53.327114], [1353945447000000.0, -6.278137, 53.325638], [1353945468000000.0, -6.2784830000000005, 53.324718000000004], [1353945487000000.0, -6.279058, 53.322952], [1353945507000000.0, -6.279381, 53.322109], [1353945510000000.0, -6.279381, 53.322109], [1353945529000000.0, -6.279369, 53.321941], [1353945547000000.0, -6.279059999999999, 53.320907999999996], [1353945570000000.0, -6.278901, 53.320576], [1353945586000000.0, -6.27877, 53.319786], [1353945607000000.0, -6.279391, 53.318249], [1353945629000000.0, -6.279657, 53.317859999999996], [1353945648000000.0, -6.2810690000000005, 53.316212], [1353945669000000.0, -6.2816529999999995, 53.31559], [1353945688000000.0, -6.2823970000000005, 53.313877000000005], [1353945709000000.0, -6.282494, 53.312435], [1353945752000000.0, -6.2831339999999996, 53.310759999999995], [1353945774000000.0, -6.2831339999999996, 53.310759999999995], [1353945791000000.0, -6.283291, 53.310390000000005], [1353945811000000.0, -6.283375, 53.310204000000006], [1353945832000000.0, -6.283976999999999, 53.308643000000004], [1353945852000000.0, -6.283989, 53.308025], [1353945873000000.0, -6.283868, 53.307460999999996], [1353945891000000.0, -6.283868, 53.307460999999996], [1353945912000000.0, -6.283868, 53.307460999999996], [1353945930000000.0, -6.2837309999999995, 53.30690799999999], [1353945951000000.0, -6.283509, 53.30616], [1353945972000000.0, -6.283373, 53.305682999999995], [1353945991000000.0, -6.283368, 53.305649], [1353946012000000.0, -6.283365, 53.304562], [1353946032000000.0, -6.283915, 53.302696], [1353946051000000.0, -6.2839339999999995, 53.301662], [1353946072000000.0, -6.283942, 53.301334], [1353946093000000.0, -6.283945, 53.301163], [1353946111000000.0, -6.283812999999999, 53.299389], [1353946131000000.0, -6.284544, 53.298237], [1353946152000000.0, -6.284769, 53.297878000000004], [1353946171000000.0, -6.28492, 53.297630000000005], [1353946191000000.0, -6.28492, 53.297630000000005], [1353946214000000.0, -6.284857, 53.297245], [1353946231000000.0, -6.284885, 53.297706999999996], [1353946274000000.0, -6.281703, 53.294993999999996], [1353946292000000.0, -6.280853, 53.294456000000004], [1353946312000000.0, -6.281636, 53.293918999999995], [1353946332000000.0, -6.282267, 53.293048999999996], [1353946351000000.0, -6.282389, 53.291821], [1353946374000000.0, -6.282564, 53.290732999999996], [1353946391000000.0, -6.28263, 53.290287], [1353946411000000.0, -6.282806, 53.289104], [1353946431000000.0, -6.283044, 53.28758199999999], [1353946450000000.0, -6.282621, 53.286880000000004], [1353946528000000.0, -6.281216000000001, 53.28476], [1353946530000000.0, -6.278632, 53.281746], [1353946572000000.0, -6.271542, 53.279644], [1353946591000000.0, -6.2704629999999995, 53.279324], [1353946610000000.0, -6.269385, 53.278969], [1353946633000000.0, -6.267348, 53.278118000000006], [1353946652000000.0, -6.265795, 53.277283], [1353946672000000.0, -6.265484, 53.27709599999999], [1353946691000000.0, -6.2654510000000005, 53.277077], [1353946712000000.0, -6.264346, 53.276286999999996], [1353946733000000.0, -6.263719999999999, 53.275753], [1353946751000000.0, -6.2629470000000005, 53.275093000000005], [1353946772000000.0, -6.261573, 53.274273], [1353946790000000.0, -6.26001, 53.273559999999996], [1353946811000000.0, -6.259355, 53.273289], [1353946834000000.0, -6.256252, 53.272282], [1353946872000000.0, -6.2516050000000005, 53.27170600000001], [1353946890000000.0, -6.249516000000001, 53.271751], [1353946911000000.0, -6.249274, 53.271758999999996], [1353946931000000.0, -6.249274, 53.271758999999996], [1353946971000000.0, -6.249274, 53.271758999999996], [1353947009000000.0, -6.24825, 53.271816], [1353947012000000.0, -6.24825, 53.271831999999996], [1353947030000000.0, -6.24825, 53.271831999999996], [1353947053000000.0, -6.24825, 53.271831999999996], [1353947068000000.0, -6.24825, 53.271831999999996], [1353947072000000.0, -6.24825, 53.271851], [1353947091000000.0, -6.24825, 53.271851], [1353947111000000.0, -6.248133, 53.271831999999996], [1353947130000000.0, -6.248083, 53.271831999999996], [1353947150000000.0, -6.248083, 53.271851], [1353947171000000.0, -6.248083, 53.271851], [1353947189000000.0, -6.248083, 53.271851], [1353947192000000.0, -6.248083, 53.271831999999996], [1353947211000000.0, -6.248083, 53.271851], [1353947231000000.0, -6.248067, 53.271851], [1353947250000000.0, -6.248067, 53.271851], [1353947252000000.0, -6.248083, 53.271851], [1353947273000000.0, -6.248083, 53.271851], [1353947292000000.0, -6.248083, 53.271851], [1353947308000000.0, -6.248083, 53.271851], [1353947310000000.0, -6.248083, 53.271851], [1353947331000000.0, -6.248083, 53.271851], [1353947351000000.0, -6.248083, 53.271851], [1353947369000000.0, -6.248083, 53.271851], [1353947373000000.0, -6.248067, 53.271851], [1353947390000000.0, -6.248067, 53.271831999999996], [1353947410000000.0, -6.248083, 53.271831999999996], [1353947429000000.0, -6.248083, 53.271831999999996], [1353947433000000.0, -6.248067, 53.271831999999996], [1353947451000000.0, -6.248067, 53.271831999999996], [1353947472000000.0, -6.248067, 53.271831999999996], [1353947491000000.0, -6.248067, 53.271831999999996], [1353947493000000.0, -6.248067, 53.271831999999996], [1353947511000000.0, -6.248067, 53.271831999999996], [1353947532000000.0, -6.248067, 53.271851], [1353947548000000.0, -6.248067, 53.271851], [1353947551000000.0, -6.248083, 53.271851], [1353947572000000.0, -6.248067, 53.271851], [1353947594000000.0, -6.248067, 53.271831999999996], [1353947609000000.0, -6.248067, 53.271831999999996], [1353947611000000.0, -6.248067, 53.271831999999996], [1353947632000000.0, -6.248067, 53.271831999999996], [1353947650000000.0, -6.248067, 53.271851], [1353947671000000.0, -6.248083, 53.271851], [1353947694000000.0, -6.248083, 53.271851], [1353947712000000.0, -6.248083, 53.271851], [1353947754000000.0, -6.247883, 53.271851], [1353947771000000.0, -6.247883, 53.271851], [1353947788000000.0, -6.247883, 53.271851], [1353947792000000.0, -6.247883, 53.271851], [1353947810000000.0, -6.247883, 53.271851], [1353947831000000.0, -6.247883, 53.271851], [1353947849000000.0, -6.247883, 53.271851], [1353947851000000.0, -6.247883, 53.271831999999996], [1353947870000000.0, -6.247883, 53.271831999999996], [1353947890000000.0, -6.247883, 53.271851], [1353947909000000.0, -6.247883, 53.271851], [1353947913000000.0, -6.247883, 53.271831999999996], [1353947932000000.0, -6.247883, 53.271851], [1353947952000000.0, -6.247883, 53.271851], [1353947969000000.0, -6.247883, 53.271851], [1353947973000000.0, -6.247883, 53.271851], [1353947991000000.0, -6.2479, 53.271851], [1353948012000000.0, -6.2479, 53.271851], [1353948029000000.0, -6.2479, 53.271851], [1353948030000000.0, -6.247883, 53.271851], [1353948051000000.0, -6.247883, 53.271851], [1353948072000000.0, -6.247883, 53.271851], [1353948088000000.0, -6.247883, 53.271851], [1353948111000000.0, -6.247883, 53.271831999999996], [1353948132000000.0, -6.247883, 53.271851], [1353948151000000.0, -6.247883, 53.271851], [1353948171000000.0, -6.2479, 53.271851], [1353948194000000.0, -6.2479, 53.271851], [1353948208000000.0, -6.2479, 53.271851], [1353948211000000.0, -6.2479, 53.271851], [1353948232000000.0, -6.247883, 53.271851], [1353948251000000.0, -6.247883, 53.271851], [1353948269000000.0, -6.247883, 53.271851], [1353948271000000.0, -6.247883, 53.271866], [1353948293000000.0, -6.247883, 53.271866], [1353948311000000.0, -6.247883, 53.271866], [1353948330000000.0, -6.247883, 53.271866], [1353948332000000.0, -6.247883, 53.271866], [1353948353000000.0, -6.247883, 53.271851], [1353948372000000.0, -6.247866999999999, 53.271851], [1353948390000000.0, -6.247883, 53.271831999999996], [1353948413000000.0, -6.247883, 53.271831999999996], [1353948432000000.0, -6.247883, 53.271831999999996], [1353948452000000.0, -6.247883, 53.271851], [1353948471000000.0, -6.2479, 53.271851], [1353948530000000.0, -6.2479, 53.271851], [1353948551000000.0, -6.2479, 53.271851], [1353948571000000.0, -6.2479, 53.271851], [1353948590000000.0, -6.247883, 53.271851], [1353948610000000.0, -6.247883, 53.271851], [1353948629000000.0, -6.247883, 53.271851], [1353948633000000.0, -6.247883, 53.271851], [1353948652000000.0, -6.247883, 53.271851], [1353948670000000.0, -6.247883, 53.271851], [1353948691000000.0, -6.247883, 53.271851], [1353948693000000.0, -6.247883, 53.271851], [1353948712000000.0, -6.247883, 53.271851], [1353948730000000.0, -6.247883, 53.271866], [1353948751000000.0, -6.247883, 53.271866], [1353948755000000.0, -6.247883, 53.271866], [1353948769000000.0, -6.2479, 53.271866], [1353948792000000.0, -6.247883, 53.271866], [1353948810000000.0, -6.247883, 53.271866], [1353948831000000.0, -6.247883, 53.271866], [1353948851000000.0, -6.247883, 53.271866], [1353948871000000.0, -6.247883, 53.271851], [1353948891000000.0, -6.247883, 53.271851], [1353948913000000.0, -6.247883, 53.271851], [1353948930000000.0, -6.247883, 53.271851], [1353948932000000.0, -6.247883, 53.271866], [1353948950000000.0, -6.247883, 53.271866], [1353948973000000.0, -6.247883, 53.271866], [1353948989000000.0, -6.247883, 53.271866]]

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
	#buffer = BytesIO(urllib.urlopen(url).read())
	#image = Image.open(buffer)
	#image.save("map{}.png".format(row[0]))
	print "Created image for Tripid:" + str(row[0])
	if len(printed_ids) == 5:
		break
