
# coding: utf-8

# In[8]:


import numpy as np
import capo
import matplotlib.pyplot as plt
import glob


# In[9]:
def avgfreqcalc(data_dir, antstr, stokes):
    xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
    xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
    yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
    yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))


    ant_i, ant_j = map(int, antstr.split('_'))




    # initialize average power
    avg_freq = None
    n_avg = 0




    # loop over files
    for i in np.arange(len(xx_data)):
        t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx')
        #t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy')
        #t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx')
        t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy')

        vis_xx = d_xx[(ant_i, ant_j)]['xx']
        vis_yy = d_yy[(ant_i, ant_j)]['yy']

        if avg_freq is None:
            avg_freq = np.zeros((vis_xx.shape[1]))
        if stokes == "I":
            stokes_I = vis_xx + vis_yy
            for it in range(vis_xx.shape[0]):    
                avg_freq += np.abs(stokes_I[it, :])
                n_avg += 1
        if stokes == "Q":
            stokes_Q = vis_xx - vis_yy
            for it in range(vis_xx.shape[0]):    
                avg_freq += np.abs(stokes_Q[it, :])
                n_avg += 1



    # finish averaging
    avg_freq = avg_freq/n_avg
    return avg_freq



    # plot the result
    # plt.plot(avg_freq)
    # plt.title("Average Stokes I over time")
    # plt.xlabel("Frequency channel")
    # plt.ylabel("Average power")
    # plt.show()
