import numpy as np
import capo
import matplotlib.pyplot as plt
import glob

def avgfreqcalc(data_dir, antstr, stokes):
    xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
    #xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
    #yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
    yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))

    ant_i, ant_j = map(int, antstr.split('_'))

    # initialize average power
    avg_freq = None
    n_avg = 0
    # loop over files
    if stokes == "I" or "Q":
        for i in np.arange(len(xx_data)):
            t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx', verbose=True)
            #t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy')
            #t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx')
            t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy', verbose=True)

            vis_xx = d_xx[(ant_i, ant_j)]['xx']
            vis_yy = d_yy[(ant_i, ant_j)]['yy']
            channels = vis_xx.shape[1]
            if avg_freq is None:
                avg_freq = np.zeros((vis_xx.shape[1]))
            if stokes == "I":
                stokes_I = vis_xx + vis_yy
                for it in range(vis_xx.shape[0]):    
                    avg_freq += np.abs(stokes_I[it, :])
                    n_avg += 1
            elif stokes == "Q":
                stokes_Q = vis_xx - vis_yy
                for it in range(vis_xx.shape[0]):    
                    avg_freq += np.abs(stokes_Q[it, :])
                    n_avg += 1
        elif stokes == "U" or "V":
            t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy')
            t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx')
            vis_xy = d_xy[(ant_i, ant_j)]['xy']
            vis_yx = d_yx[(ant_i, ant_j)]['yx']
            channels = vis_xy.shape[1]
            if avg_freq is None:
                avg_freq = np.zeros((vis_xx.shape[1]))
            if stokes == "U":
                stokes_U = vis_xy + vis_yx
                for it in range(vis_xy.shape[0]):    
                    avg_freq += np.abs(stokes_U[it, :])
                    n_avg += 1
            elif stokes == "V":
                stokes_V = np.imag(vis_xy) - np.imag(vis_yx)
                for it in range(vis_xy.shape[0]):    
                    avg_freq += np.abs(stokes_U[it, :])
                    n_avg += 1
        else:
            print ("I'm sorry, but this script does not yet support the stokes you have requested.")

    # finish averaging
    avg_freq = avg_freq/n_avg
    return avg_freq, channels



    # plot the result
    # plt.plot(avg_freq)
    # plt.title("Average Stokes I over time")
    # plt.xlabel("Frequency channel")
    # plt.ylabel("Average power")
    # plt.show()
def avgfreqloop(data_dir, stokes):
    baselines = ['64_88', '64_80', '9_105', '9_53', '53_104', '22_72', '20_22', '20_31', '31_96', '65_89', '10_97', '10_43', '72_105', '88_105', '22_112', '9_22', '9_64', '20_53', '53_80', '10_89', '31_89', '31_104', '43_65', '65_96', '72_112', '97_112', '22_105', '9_88', '9_20', '20_89', '43_89', '53_64', '31_53', '31_65', '80_104', '96_104']
    for antstr in baselines:
        ant_i, ant_j = map(int, antstr.split('_'))
        if stokes == "I":
            xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
            yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))
            for i in np.arange(len(xx_data)):
                t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx')
                t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy')
                vis_xx = d_xx[(ant_i, ant_j)]['xx']
                vis_yy = d_yy[(ant_i, ant_j)]['yy']

    xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
    #xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
    #yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
    yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))

