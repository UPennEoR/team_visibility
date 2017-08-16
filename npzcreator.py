from __future__ import print_function, division
import glob
import numpy as np
import capo
import matplotlib.pyplot as plt
import imageio
import os
import hsa7458_v001 as cal
from operator import itemgetter
import time




def calculate_baseline(antennae, pair):
    """
    The decimal module is necessary for keeping the number of decimal places small.
    Due to small imprecision, if more than 8 or 9 decimal places are used, 
    many baselines will be calculated that are within ~1 nanometer to ~1 picometer of each other.
    Because HERA's position precision is down to the centimeter, there is no 
    need to worry about smaller imprecision.
    """

    dx = antennae[pair[0]]['top_x'] - antennae[pair[1]]['top_x']
    dy = antennae[pair[0]]['top_y'] - antennae[pair[1]]['top_y']
    baseline = np.around([np.sqrt(dx**2. + dy**2.)] , 3)[0] #XXX this may need tuning
    slope = dy/np.float64(dx)
    if slope == -np.inf:
        slope = slope * -1
    elif slope == 0:
        slope = slope + 0
    ps = (pair[0],pair[1],"%.2f" % slope)
    return "%.1f" % baseline,ps

ex_ants = [72,81]
antennae = cal.prms['antpos_ideal']
baselines = {}

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

        baseline, ps = calculate_baseline(antennae, pair)

        if (baseline not in baselines):
            baselines[baseline] = [ps]
        elif (pair in baselines[baseline]):
            continue
        else:
            baselines[baseline].append(ps)


data_dir = '/data4/paper/HERA2015/2457555/PennData/RFI_flag/'
my_path = '/data4/paper/rkb/NPZstorage/'
keys = sorted(baselines) 

times = [3,4,5]
xx_data = []
xy_data = []
yx_data = []
yy_data = []

t0 = time.time()

for i in times: 
    xx_data += sorted(glob.glob(''.join([data_dir, 'zen.*.{}*.xx.HH.uvcR'.format(i)])))
    xy_data += sorted(glob.glob(''.join([data_dir, 'zen.*.{}*.xy.HH.uvcR'.format(i)])))
    yx_data += sorted(glob.glob(''.join([data_dir, 'zen.*.{}*.yx.HH.uvcR'.format(i)])))
    yy_data += sorted(glob.glob(''.join([data_dir, 'zen.*.{}*.yy.HH.uvcR'.format(i)])))


antlist_all = []
for i in keys:
    x = sorted(set(baselines[i]), key=itemgetter(2))

    for elem,antstr in enumerate(x):
        antlist_all.append("%s_%s" % (x[elem][0], x[elem][1]))

    antstr_all = ','.join(antlist_all)

n_avg = 0

avgvis_dict = {}

for i in range(len(xx_data)):
    
    #print("Reading {}...".format(xx_data[i]))
    t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr_all, polstr='xx')
    #print("Reading {}...".format(xy_data[i]))
    t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr_all, polstr='xy')
    #print("Reading {}...".format(yx_data[i]))
    t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr_all, polstr='yx')
    print("Reading {}...".format(yy_data[i]))
    t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr_all, polstr='yy')

    for elem,antstr in enumerate(antlist_all):
        ant_i, ant_j = map(int, antstr.split('_'))

        vis_xx = d_xx[(ant_i, ant_j)]['xx']
        vis_yy = d_yy[(ant_i, ant_j)]['yy']
        vis_yx = d_yx[(ant_i, ant_j)]['yx']
        vis_xy = d_xy[(ant_i, ant_j)]['xy']


        


        vis_xx_real = vis_xx.real
        vis_xx_imag = vis_xx.imag
        vis_yy_real = vis_yy.real
        vis_yy_imag = vis_yy.imag
        vis_yx_real = vis_yx.real
        vis_yx_imag = vis_yx.imag
        vis_xy_real = vis_xy.real
        vis_xy_imag = vis_xy.imag

        if ('%s' %(antstr) not in avgvis_dict):
            avgvis_dict['%s' %(antstr)]={}

            avgvis_dict['%s' %(antstr)]['xx_real'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['xx_imag'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['yy_real'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['yy_imag'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['yx_real'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['yx_imag'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['xy_real'] = np.zeros((vis_xx.shape[1]))
            avgvis_dict['%s' %(antstr)]['xy_imag'] = np.zeros((vis_xx.shape[1]))

        for it in range(vis_xx.shape[0]):
            avgvis_dict['%s' %(antstr)]['xx_real'] += vis_xx_real[it,:]
            avgvis_dict['%s' %(antstr)]['xx_imag'] += vis_xx_imag[it,:]
            avgvis_dict['%s' %(antstr)]['yy_real'] += vis_yy_real[it,:]
            avgvis_dict['%s' %(antstr)]['yy_imag'] += vis_yy_imag[it,:]
            avgvis_dict['%s' %(antstr)]['yx_real'] += vis_yx_real[it,:]
            avgvis_dict['%s' %(antstr)]['yx_imag'] += vis_yx_imag[it,:]
            avgvis_dict['%s' %(antstr)]['xy_real'] += vis_xy_real[it,:]
            avgvis_dict['%s' %(antstr)]['xy_imag'] += vis_xy_imag[it,:]
            n_avg += 1

print (antlist_all)
for elem,antstr in enumerate(antlist_all):    
    avgvis_dict['%s' %(antstr)]['xx_real'] /= n_avg
    avgvis_dict['%s' %(antstr)]['xx_imag'] /= n_avg
    avgvis_dict['%s' %(antstr)]['yy_real'] /= n_avg
    avgvis_dict['%s' %(antstr)]['yy_imag'] /= n_avg
    avgvis_dict['%s' %(antstr)]['yx_real'] /= n_avg
    avgvis_dict['%s' %(antstr)]['yx_imag'] /= n_avg
    avgvis_dict['%s' %(antstr)]['xy_real'] /= n_avg
    avgvis_dict['%s' %(antstr)]['xy_imag'] /= n_avg

np.savez(my_path+'2457755.RFIraw.avgvis.npz',
         avgvis_dict = avgvis_dict)
t1 = time.time()

total = t1-t0
print (total,"secs")