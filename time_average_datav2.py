# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import os
import re
import matplotlib
import glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyuvdata import UVData

make_files = True
plot_files = True

# save total number of time samples
# 56*71 + 45 (last file is shorter) = 4021
Ntimes = 4133

# define all east-west 14 meter baselines
ew14m = [
    '9_64',
    '10_89',
    '53_80',
    '97_81',
]

def time_average_data(data_dir, outfile, file_type):
    # initialize dict for average visibilities
    avg_vis = {}
    counts = {}

    # loop over files and add to average
    xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcRO'])))
    for fn in xxdatafiles:
        print("Reading {}".format(fn))
        uvd = UVData()
        uvd.read_miriad(fn)
        # loop over all visibilities in file
        for key, d in uvd.antpairpol_iter():
            ind1, ind2, ipol = uvd._key2inds(key)
            for ind in [ind1, ind2]:
                if len(ind) == 0:
                    continue
                f = uvd.flag_array[ind, 0, :, ipol]
                dkey = '{0}_{1}'.format(key[0], key[1])
                d_flg = np.where(f, np.nan, d)
                try:
                    # add to running total
                    avg_vis[dkey] += np.nansum(d_flg, axis=0, dtype=np.complex128)
                    counts[dkey] += np.sum(np.logical_not(f), axis=0)
                except KeyError:
                    # first iteration through; initialize dictionary
                    avg_vis[dkey] = np.nansum(d_flg, axis=0, dtype=np.complex128)
                    counts[dkey] = np.sum(np.logical_not(f), axis=0)

    # compute final average
    for key in avg_vis.keys():
        avg_vis[key] = np.where(counts[key] > 0, avg_vis[key]/counts[key], 0.)

    # save out average visibilities dict and counts dict
    print("Saving {}".format(outfile))
    np.savez(outfile, avg_vis)
    outfile3 = re.sub('_', '_counts_', outfile)
    print("Saving {}".format(outfile3))
    np.savez(outfile3, counts)

    return

def plot_time_average_data(infiles, outfiles, file_type):
    # set some options
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('lines', linewidth=1)

    # make list of frequencies
    freqs = np.linspace(100, 200, num=1024, endpoint=False)

    # loop over input files
    for i, infile in enumerate(infiles):
        # make empty new figures
        print(infile)
        fig1 = plt.figure()
        ax_re = plt.gca()
        fig2 = plt.figure()
        ax_im = plt.gca()

        # load average visibility and count dictionaries
        f = np.load(infile)
        d = f['arr_0'].item()
        infile2 = re.sub('_', '_counts_', infile)
        f2 = np.load(infile2)
        counts = f2['arr_0'].item()

        # loop over all baselines defined above
        for bl in ew14m:
            try:
                avg_vis = d[bl]
                key = bl
            except KeyError:
                # antenna is not in dict; reverse antenna numbers in list
                ants = bl.split('_')
                key = '_'.join(ants[::-1])
                avg_vis = d[key]
            # mask out NaNs
            avg_vis = np.where(np.isnan(avg_vis), 0., avg_vis)
            # throw out channels that are flagged more than 20% of the time
            avg_vis = np.where(counts[key] / Ntimes < 0.8, 0., avg_vis)
            label = re.sub('_', ',', bl)
            ax_re.plot(freqs, avg_vis.real, label=label, alpha=0.5)
            ax_im.plot(freqs, avg_vis.imag, linestyle='--', label=label, alpha=0.5)

        # make plot pretty
        ax_re.set_xlabel('Freq [MHz]')
        ax_im.set_xlabel('Freq [MHz]')
        ax_re.set_xlim((100,200))
        ax_im.set_xlim((100,200))
        leg = ax_re.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg = ax_im.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fn_re = ''.join([outfiles[i], '_re.pdf'])
        fn_im = ''.join([outfiles[i], '_im.pdf'])
        print("Saving {}".format(fn_re))
        fig1.savefig('/data4/paper/rkb/miriadtimeavgreaderstorage/XXRFI/down1left1oneout.png', bbox_inches='tight')

    return

if __name__ == '__main__':
    if make_files:
        xx_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.xx.HH.uvc{}'.format(file_type) in fn]
        time_average_data(xx_files, 'xx_uvc{}.npz'.format(file_type))
        xy_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.xy.HH.uvc{}'.format(file_type) in fn]
        time_average_data(xy_files, 'xy_uvc{}.npz'.format(file_type))
        yx_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.yx.HH.uvc{}'.format(file_type) in fn]
        time_average_data(yx_files, 'yx_uvc{}.npz'.format(file_type))
        yy_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.yy.HH.uvc{}'.format(file_type) in fn]
        time_average_data(yy_files, 'yy_uvc{}.npz'.format(file_type))
    if plot_files:
        infiles = ['xx_uvc{}.npz'.format(file_type),
                   'xy_uvc{}.npz'.format(file_type),
                   'yx_uvc{}.npz'.format(file_type),
                   'yy_uvc{}.npz'.format(file_type)]
        outfiles = ['avg_vis_{}_xx'.format(file_type),
                    'avg_vis_{}_xy'.format(file_type),
                    'avg_vis_{}_yx'.format(file_type),
                    'avg_vis_{}_yy'.format(file_type)]
        plot_time_average_data(infiles, outfiles)
