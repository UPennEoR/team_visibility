
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import itertools
import capo
import glob
import matplotlib.pyplot as plt
#%matplotlib inline
#abc rules:
#use a before b, b before c
#only use two vectors in describing path


# In[2]:


#all 1a baselines
antstr1 = ['72_112','97_112','22_105','22_81','10_81','9_88', '9_20','20_89','43_89','53_64','31_53','31_65','10_80','10_96']


# In[4]:


avg_freq = None
n_avg = 0
for i in antstr1:
    ant_i, ant_j = map(int, i.split('_'))
    t_xx, d_xx, f_xx = capo.miriad.read_files(['/data4/paper/HERA2015/2457746/zen.2457746.16693.xx.HH.uvcORR'], antstr=i, polstr='xx')
    t_xy, d_xy, f_xy = capo.miriad.read_files(['/data4/paper/HERA2015/2457746/zen.2457746.16693.xy.HH.uvcORR'], antstr=i, polstr='xy')
    t_yx, d_yx, f_yx = capo.miriad.read_files(['/data4/paper/HERA2015/2457746/zen.2457746.16693.yx.HH.uvcORR'], antstr=i, polstr='yx')
    t_yy, d_yy, f_yy = capo.miriad.read_files(['/data4/paper/HERA2015/2457746/zen.2457746.16693.yy.HH.uvcORR'], antstr=i, polstr='yy')
    d_xx = (d_xx[(ant_i, ant_j)]['xx'])
    #print (d_xx)
    xx = np.zeros(d_xx.shape)
    xx += np.real(d_xx)
    d_xy = (d_xy[(ant_i, ant_j)]['xy'])
    xy = np.zeros(d_xy.shape)
    xy += np.real(d_xy)
    d_yx = (d_yx[(ant_i, ant_j)]['yx'])
    yx = np.zeros(d_yx.shape)
    yx += np.real(d_yx)
    d_yy = (d_yy[(ant_i, ant_j)]['yy'])
    yy = np.zeros(d_yy.shape)
    yy += np.real(d_yy)
if avg_freq is None:
    avg_freq = np.zeros((xx.shape[1]))

stokes_I = xx + yy
for it in range(xx.shape[0]):
    avg_freq += np.abs(stokes_I[it, :])
    n_avg += 1
p = len(antstr1)
avg_freq = avg_freq/p
print (avg_freq)
#plt.imshow((avg_stokes),aspect='auto',vmax=2
plt.plot(avg_freq)
# plt.xlabel('Frequency Channel')
# plt.ylabel('Average Power')
#plt.show()
plt.savefig('delay.png')
# avg_xx = xx/p
# avg_yx = yx/p
# avg_xy = xy/p
# avg_yy = yy/p
# #plt.imshow((avg_yx), aspect='auto', vmax=20, vmin=-20, cmap='jet')


# In[ ]:





# In[ ]:




