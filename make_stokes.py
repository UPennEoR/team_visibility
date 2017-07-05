import numpy as np
import capo
import matplotlib.pyplot as plt

antstr = '97_112'
ant_i, ant_j = map(int, antstr.split('_'))

t_xx, d_xx, f_xx = capo.miriad.read_files(['/Users/plaplant/Documents/school/penn/2457700/zen.2457700.35483.xx.HH.uvc'], antstr=antstr, polstr='xx')
t_xy, d_xy, f_xy = capo.miriad.read_files(['/Users/plaplant/Documents/school/penn/2457700/zen.2457700.35483.xy.HH.uvc'], antstr=antstr, polstr='xy')
t_yx, d_yx, f_yx = capo.miriad.read_files(['/Users/plaplant/Documents/school/penn/2457700/zen.2457700.35483.yx.HH.uvc'], antstr=antstr, polstr='yx')
t_yy, d_yy, f_yy = capo.miriad.read_files(['/Users/plaplant/Documents/school/penn/2457700/zen.2457700.35483.yy.HH.uvc'], antstr=antstr, polstr='yy')

plt.subplot(241)
plt.imshow(np.log10(np.abs(d_xx[(ant_i, ant_j)]['xx'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('xx Visibility')
plt.xlabel('Frequency bin')
plt.ylabel('LST')
plt.subplot(242)
plt.imshow(np.log10(np.abs(d_xy[(ant_i, ant_j)]['xy'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('xy Visibility')
plt.xlabel('Frequency bin')
plt.ylabel('LST')
plt.subplot(243)
plt.imshow(np.log10(np.abs(d_yx[(ant_i, ant_j)]['yx'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('yx Visibility')
plt.xlabel('Frequency bin')
plt.ylabel('LST')
plt.subplot(244)
plt.imshow(np.log10(np.abs(d_yy[(ant_i, ant_j)]['yy'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('yy Visibility')
plt.xlabel('Frequency bin')
plt.ylabel('LST')

vis_xx = d_xx[(ant_i, ant_j)]['xx']
vis_yy = d_yy[(ant_i, ant_j)]['yy']
vis_yx = d_yx[(ant_i, ant_j)]['yx']
vis_xy = d_xy[(ant_i, ant_j)]['xy']

# make pseudo-Stokes visibilities
stokes_I = vis_xx + vis_yy
stokes_Q = vis_xx - vis_yy
stokes_U = vis_xy + vis_yx
stokes_V = 1j*vis_xy - 1j*vis_yx

plt.subplot(245)
plt.imshow(np.log10(np.abs(stokes_I)), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('Stokes I')
plt.xlabel('Frequency bin')
plt.ylabel('LST')

plt.subplot(246)
plt.imshow(np.log10(np.abs(stokes_Q)), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('Stokes Q')
plt.xlabel('Frequency bin')
plt.ylabel('LST')

plt.subplot(247)
plt.imshow(np.log10(np.abs(stokes_U)), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('Stokes U')
plt.xlabel('Frequency bin')
plt.ylabel('LST')

plt.subplot(248)
plt.imshow(np.log10(np.abs(stokes_V)), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
plt.title('Stokes V')
plt.xlabel('Frequency bin')
plt.ylabel('LST')

#plt.show()
plt.savefig('{}.png'.format(antstr))
