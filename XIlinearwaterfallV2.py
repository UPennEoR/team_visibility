import numpy as np
import capo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time

# julian date of observation
jd = 2457555.35484

# latitude and longitude of HERA in degrees
latitude = -30.2715
longitude = 21.428

t = Time(jd, format='jd', location=(longitude, latitude))

lst = t.sidereal_time('apparent')
print lst

antstr = ['9_20']

for elem,antstr1 in enumerate(antstr):
    
    ant_i, ant_j = map(int, antstr[elem].split('_'))
    ##print ant_i , ant_j
    
    t_xx, d_xx, f_xx = capo.miriad.read_files(['/data4/paper/HERA2015/2457555/PennData/RFI_flag/zen.2457555.35484.xx.HH.uvcR'], antstr=antstr1, polstr='xx')
    t_xy, d_yy, f_xy = capo.miriad.read_files(['/data4/paper/HERA2015/2457555/PennData/RFI_flag/zen.2457555.35484.yy.HH.uvcR'], antstr=antstr1, polstr='yy')
    
    
    vis_xx = d_xx[(ant_i, ant_j)]['xx']
    vis_yy = d_yy[(ant_i, ant_j)]['yy']

    stokes_I = vis_xx + vis_yy
    
    x = range(56)
    labels = [''] * 56
    labels[0] = "1h06m32s"
    labels[14]= "1h08m52s"
    labels[28]= "1h011m11s"
    labels[42]= "1h013m30s"
    labels[55]= "1h015m49s"
    
    plt.figure(figsize=(12, 7))
    G = gridspec.GridSpec(16, 4)

    axes_1 = plt.subplot(G[:14, 0:2])
    c = plt.imshow(np.log10(np.abs(d_xx[(ant_i, ant_j)]['xx'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
    plt.title('Visibility Amplitude %s'%(antstr1),fontsize = 20)
    plt.xlabel('Frequency bin',fontsize = 14)
    plt.ylabel('LST',fontsize = 14)
    plt.yticks(x, labels)

    axes_2 = plt.subplot(G[:14, 2:4])
    r = plt.imshow(np.angle(stokes_I), aspect='auto', vmax=np.pi, vmin=-np.pi, cmap='viridis')
    plt.title('Visibility Phase %s'%(antstr1),fontsize = 20)
    plt.xlabel('Frequency bin',fontsize = 14)
    plt.ylabel('LST',fontsize = 14)
    plt.yticks(x, labels)

    axes_3 = plt.subplot(G[15, 0:2])
    plt.colorbar(c,cax=axes_3,orientation='horizontal')
    plt.xlabel('spectral power flux density',fontsize = 14)
    
    axes_4 = plt.subplot(G[15, 2:4])
    plt.colorbar(r,cax=axes_4,orientation='horizontal')
    plt.xlabel('spectral power flux density',fontsize = 14)

    plt.tight_layout()
    plt.savefig('2457755.RFIraw.visI_t{}.pdf'.format(antstr1))
    plt.close()
