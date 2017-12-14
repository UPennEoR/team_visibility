# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyuvdata import UVData

make_files = True
plot_files = False

# min and max LST in radians
lst_min = 2.7749
lst_max = 5.9225

# define all east-west 14 meter baselines
ew14m = [
    '72_112',
    '112_97',
    '105_22',
    '22_81',
    '81_10',
    '88_9',
    '9_20',
    '20_89',
    '89_43',
    '64_53',
    '53_31',
    '31_65',
    '80_104',
    '104_96',
]

def time_average_data(file_list, outfile):
    # initialize dict for average visibilities
    avg_vis = {}
    counts = {}

    # loop over files and add to average
    for fn in file_list:
        print("Reading {}".format(fn))
        uvd = UVData()
        uvd.read_miriad(fn)

        # get indices corresponding to values inside the LST range
        lst_ind = np.logical_and(uvd.lst_array > lst_min, uvd.lst_array < lst_max)

        # loop over all visibilities in file
        for key, d in uvd.antpairpol_iter():
            ind1, ind2, ipol = uvd._key2inds(key)
            for ind in [ind1, ind2]:
                # protect against empty indices
                if len(ind) == 0:
                    continue

                # mask out indices not in the specified LST range
                lind = np.where(lst_ind[ind])
                ind = ind[lind]

                # skip empty waterfalls
                if len(ind) == 0:
                    continue

                # grab only the data that made it through
                d = uvd.data_array[ind, 0, :, ipol]
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

def plot_time_average_data(infiles, outfiles):
    # set some options
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('lines', linewidth=1)

    # make list of frequencies
    freqs = np.linspace(100, 200, num=1024, endpoint=False)

    # loop over input files
    for i, infile in enumerate(infiles):
        # make empty new figures
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
        fig1.savefig(fn_re, bbox_inches='tight')
        print("Saving {}".format(fn_im))
        fig2.savefig(fn_im, bbox_inches='tight')

    return

if __name__ == '__main__':
    if make_files:
        xx_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.xx.HH.uv' in fn]
        time_average_data(xx_files, 'xx_uv.npz')
        xy_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.xy.HH.uv' in fn]
        time_average_data(xy_files, 'xy_uv.npz')
        yx_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.yx.HH.uv' in fn]
        time_average_data(yx_files, 'yx_uv.npz')
        yy_files = [fn for fn in sorted(os.listdir(os.getcwd())) if '.yy.HH.uv' in fn]
        time_average_data(yy_files, 'yy_uv.npz')
    if plot_files:
        infiles = ['xx_uv.npz',
                   'xy_uv.npz',
                   'yx_uv.npz',
                   'yy_uv.npz']
        outfiles = ['avg_vis_xx',
                    'avg_vis_xy',
                    'avg_vis_yx',
                    'avg_vis_yy']
        plot_time_average_data(infiles, outfiles)
