import capo
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
import aipy

def calculate_baseline(antennae, pair):
	dx = antennae[pair[0]]['top_x'] - antennae[pair[1]]['top_x']
	dy = antennae[pair[0]]['top_y'] - antennae[pair[1]]['top_y']
	baseline = np.around([np.sqrt(dx**2. + dy**2.)],3)[0] #XXX this may need tuning
	return baseline

def calculate_angle(antennae, pair):
	dx = antennae[pair[0]]['top_x'] - antennae[pair[1]]['top_x']
	dy = antennae[pair[0]]['top_y'] - antennae[pair[1]]['top_y']
	angle = np.around([np.tan(dx/dy)],3)[0]
	return angle
"""
Returns a dictionary of redundant baselines based on a calfile, and
excluding bad antennae, specified by ex_ants, a list of integers.

Requires cal file to be in PYTHONPATH.
"""
calfile = f.open("/data4/paper/rkb/hsa7458_v001.py")
try:
	print 'reading, %s'%calfile
	exec("import {cfile} as cal".format(cfile=calfile))
	antennae = cal.prms['antpos_ideal']
except ImportError:
	raise Exception("Unable to import {cfile}.".format(cfile=calfile))
f.close()
"""
determines the baseline and places them in the dictionary.
excludes antennae with z-position < 0 or if in ex_ants list
"""
baselines = {}
angles = {}
for antenna_i in antennae:
	if antennae[antenna_i]['top_z'] < 0.:
		continue
	if antenna_i in ex_ants:
		continue
	
	for antenna_j in antennae:
		if antennae[antenna_j]['top_z'] < 0.:
			continue
		if antenna_j in ex_ants:
			continue

		if antenna_i == antenna_j:
			continue
		elif antenna_i < antenna_j:
			pair = (antenna_i, antenna_j)
		elif antenna_i > antenna_j:
			pair = (antenna_j, antenna_i)

		baseline = calculate_baseline(antennae, pair)
		angle = calculate_angle(antennae, pair)
		
		if (baseline not in baselines):
			baselines[baseline] = [pair]
		elif (pair in baselines[baseline]):
			continue
		else:
			baselines[baseline].append(pair)
		if (angle not in angles):
			angles[angle]= [pair]
		elif (pair in angles[angle]):
			continue
		else:
			angles[angle].append(pair)
return baselines
#print (angles)

#def order_baselines(baselines, angles):
	#orders baselines by baseline primarily, and then by angle. E.x. if 
	#two pairs were 2 meters apart, but one pair had an angle of 1 radian wheras the other
	#was 3 radians, the one with the 1 radian would be listed first.


