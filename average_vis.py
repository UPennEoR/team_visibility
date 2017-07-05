from __future__ import print_function, division
import numpy as np
import capo
import matplotlib.pyplot as plt
import glob

# define files to read in
data_dir = '/Users/plaplant/Documents/school/penn/2457700/'
xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvc']))
xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvc']))
yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvc']))
yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvc']))

antstr = '72_112'
ant_i, ant_j = map(int, antstr.split('_'))

# initialize average power
avg_freq = None
n_avg = 0

# loop over files
for i, filename in enumerate(xx_data):
    print("Reading {}...".format(xx_data[i]))
    t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx')
    print("Reading {}...".format(xy_data[i]))
    t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy')
    print("Reading {}...".format(yx_data[i]))
    t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx')
    print("Reading {}...".format(yy_data[i]))
    t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy')

    vis_xx = d_xx[(ant_i, ant_j)]['xx']
    vis_yy = d_yy[(ant_i, ant_j)]['yy']

    stokes_I = vis_xx + vis_yy

    if avg_freq is None:
        avg_freq = np.zeros((vis_xx.shape[1]))

    # loop over times
    for it in range(vis_xx.shape[0]):
        avg_freq += np.abs(stokes_I[it, :])
        n_avg += 1

# finish averaging
avg_freq = avg_freq/n_avg

# plot the result
plt.plot(avg_freq)
plt.title("Average Stokes I over time")
plt.xlabel("Frequency channel")
plt.ylabel("Average power")
plt.show()
