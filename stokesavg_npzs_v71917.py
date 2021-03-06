# -*- coding: utf-8 -*-

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

verbose = True

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

ex_ants = [81, 72]
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


data_dir = '/data4/paper/rkb/'
my_path = '/data4/paper/rkb/'
keys = sorted(baselines) 
t0 = time.time()

xx_data = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR'])))
xy_data = sorted(glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR'])))
yx_data = sorted(glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR'])))
yy_data = sorted(glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR'])))


# generate antstr that has all baseline pairs in it
# e.g., antstr = '72_112,81_95,72_81'
antlist_all = []
for i in keys:
    x = sorted(set(baselines[i]), key=itemgetter(2))

    for elem,antstr in enumerate(x):
        antlist_all.append("%s_%s" % (x[elem][0], x[elem][1]))

    antstr_all = ','.join(antlist_all)

n_avg = 0
avgstokes_dict = {}

for i in range(len(xx_data)):
    print(i,end=" ")
    if verbose:
        print("Reading {}...".format(xx_data[i]))
    t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr_all, polstr='xx')
    if verbose:
        print("Reading {}...".format(xy_data[i]))
    t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr_all, polstr='xy')
    if verbose:
        print("Reading {}...".format(yx_data[i]))
    t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr_all, polstr='yx')
    if verbose:
        print("Reading {}...".format(yy_data[i]))
    t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr_all, polstr='yy')

    for elem,antstr in enumerate(antlist_all):
        ant_i, ant_j = map(int, antstr.split('_'))

        vis_xx = d_xx[(ant_i, ant_j)]['xx']
        vis_yy = d_yy[(ant_i, ant_j)]['yy']
        vis_yx = d_yx[(ant_i, ant_j)]['yx']
        vis_xy = d_xy[(ant_i, ant_j)]['xy']

        stokes_I = vis_xx + vis_yy
        stokes_Q = vis_xx - vis_yy
        stokes_U = vis_xy + vis_yx
        stokes_V = 1j*vis_xy - 1j*vis_yx

        stokes_I_real = stokes_I.real
        stokes_I_imag = stokes_I.imag
        stokes_Q_real = stokes_Q.real
        stokes_Q_imag = stokes_Q.imag
        stokes_U_real = stokes_U.real
        stokes_U_imag = stokes_U.imag
        stokes_V_real = stokes_V.real
        stokes_V_imag = stokes_V.imag

        if ('%s' %(antstr) not in avgstokes_dict):
            avgstokes_dict['%s' %(antstr)]={}

            avgstokes_dict['%s' %(antstr)]['i_real'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['i_imag'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['q_real'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['q_imag'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['u_real'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['u_imag'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['v_real'] = np.zeros((vis_xx.shape[1]))
            avgstokes_dict['%s' %(antstr)]['v_imag'] = np.zeros((vis_xx.shape[1]))

        for it in range(vis_xx.shape[0]):
            avgstokes_dict['%s' %(antstr)]['i_real'] += stokes_I_real[it,:]
            avgstokes_dict['%s' %(antstr)]['i_imag'] += stokes_I_imag[it,:]
            avgstokes_dict['%s' %(antstr)]['q_real'] += stokes_Q_real[it,:]
            avgstokes_dict['%s' %(antstr)]['q_imag'] += stokes_Q_imag[it,:]
            avgstokes_dict['%s' %(antstr)]['u_real'] += stokes_U_real[it,:]
            avgstokes_dict['%s' %(antstr)]['u_imag'] += stokes_U_imag[it,:]
            avgstokes_dict['%s' %(antstr)]['v_real'] += stokes_V_real[it,:]
            avgstokes_dict['%s' %(antstr)]['v_imag'] += stokes_V_imag[it,:]
            n_avg += 1


for elem,antstr in enumerate(antlist_all):    
    avgstokes_dict['%s' %(antstr)]['i_real'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['i_imag'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['q_real'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['q_imag'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['u_real'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['u_imag'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['v_real'] /= n_avg
    avgstokes_dict['%s' %(antstr)]['v_imag'] /= n_avg

np.savez(my_path+'zen.2457746.avgstokes.npz',
         avgstokes_dict = avgstokes_dict)
t1 = time.time()

total = t1-t0
print (total,"secs")

