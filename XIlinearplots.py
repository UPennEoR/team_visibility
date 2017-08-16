import h5py
import numpy as np
import glob
import h5py
import glob
import numpy as np
import capo
import matplotlib.pyplot as plt
import imageio
import os
import hsa7458_v001 as cal
from operator import itemgetter
import matplotlib.lines as mlines


data_dir = '/data4/paper/rkb/NPZstorage/'
data = sorted(glob.glob(''.join([data_dir, '*.npz'])))
datadict = np.load(data[0])


#reading in hdf5 files.
hdf5 = '/home/plaplant/global_signal/Output/HERA/beam_zenith/xi_nu_phi_vis.hdf5'
fn = hdf5
f = h5py.File(fn, 'r')
dset_xi=f["/Data"]["xi"]
xi = np.asarray(dset_xi)


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
    baseline = np.around([np.sqrt(dx**2. + dy**2.)],3)[0] #XXX this may need tuning
    slope = dy/np.float64(dx)
    if slope == -np.inf:
        slope = slope * -1
    elif slope == 0:
        slope = slope + 0

    ps = (pair[0],pair[1],"%.2f" % slope)
    return "%.1f" % baseline,ps

ex_ants=[72, 81]
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
        
        baseline,ps = calculate_baseline(antennae, pair)
        
        if (baseline not in baselines):
            baselines[baseline] = [ps]
        elif (pair in baselines[baseline]):
            continue
        else:
            baselines[baseline].append(ps)
     




keys = sorted(baselines) 
xr = np.arange(100.,200.,100./1024)
xr1 = np.arange(100.,200.5,1./2)

xdeg = range(0,360,15)

for iq,ibs in enumerate(keys):
    
    x= sorted(set(baselines[ibs]),key=itemgetter(2))
    
    seen = set()
    [item for item in x if item[2] not in seen and not seen.add(item[2])]
    seen =  sorted(seen)
    
    testbl1deg = list(np.arctan( [float(i) for i in sorted(list(set(seen)))] ) *180 / np.pi)
    
    for q,k in zip(seen,testbl1deg):
        
        res = [k1 for k1 in x if q in k1]
        
        phi = []

        if k < 0:
            b = min(range(len(xdeg)), key=lambda i: abs(xdeg[i]-360+k))
            phi.append(b)
            b = min(range(len(xdeg)), key=lambda i: abs(xdeg[i]-(180+k) ))
            phi.append(b)
        else:
            b = min(range(len(xdeg)), key=lambda i: abs(xdeg[i]-k))
            phi.append (b)
            phi.append(min(range(len(xdeg)), key=lambda i: abs(xdeg[i]- (180+k) )))

        
        fig = plt.figure(figsize=(10,10))

        for w in phi:
            
            tvis_i = xi[iq,0,w,:]
            tvis_q = xi[iq,1,w,:]
            tvis_u = xi[iq,2,w,:]
            tvis_v = xi[iq,3,w,:]


            ax11=fig.add_subplot(411)
            ax11.plot(xr1,tvis_i.real , alpha=0.75, linewidth = 4 , linestyle = '-.')
            ax11.yaxis.set_label_position("right")
            ax11.yaxis.tick_right()
            ax11.set_ylabel('Average power',fontsize = 8)
           

            ax22=fig.add_subplot(412)
            ax22.plot(xr1,tvis_q.real , alpha=0.75, linewidth = 4, linestyle = '-.')
            ax22.yaxis.set_label_position("right")
            ax22.yaxis.tick_right()
            ax22.set_ylabel('Average power',fontsize = 8)
            

            ax33=fig.add_subplot(413)
            ax33.plot(xr1,tvis_u.real , alpha=0.75, linewidth = 4, linestyle = '-.')
            ax33.yaxis.set_label_position("right")
            ax33.yaxis.tick_right()
            ax33.set_ylabel('Average power',fontsize = 8)

            ax44=fig.add_subplot(414)
            ax44.plot(xr1,tvis_v.real , alpha=0.75, linewidth = 4, linestyle = '-.')
            ax44.yaxis.set_label_position("right")
            ax44.yaxis.tick_right()
            ax44.set_ylabel('Average power',fontsize = 8)




        for elem,antstr in enumerate(res):

            antstr1 = "%s_%s" % (res[elem][0], res[elem][1])


            qwerty = datadict['avgvis_dict']
            qwerty = qwerty.item()


            vis_i = qwerty['{}'.format(antstr1)]['xx_real']
            vis_q = qwerty['{}'.format(antstr1)]['xy_real']
            vis_u = qwerty['{}'.format(antstr1)]['yx_real']
            vis_v = qwerty['{}'.format(antstr1)]['yy_real']

            limsi = 10000*np.log10(vis_i) 
            ax1 = fig.add_subplot(411,sharex=ax11,frameon=False)
            ax1.plot(xr,vis_i,alpha=0.6)
            ax1.set_title('Vis xx bs:%s m:%s '%(ibs , q),fontsize = 10)
            ax1.set_xlabel('Frequency (MHz)',fontsize = 8)
            ax1.set_ylabel('Average power',fontsize = 8)
            #ax1.set_ylim(min(limsi[limsi != -np.inf])/2.,max(np.log10(vis_i)))

            limsq = 10000*np.log10(vis_q) 
            ax2 = fig.add_subplot(412,sharex=ax22,frameon=False)
            ax2.plot(xr,vis_q,alpha=0.6)
            ax2.set_title('Vis xy bs:%s m:%s '%(ibs , q),fontsize = 10)
            ax2.set_xlabel('Frequency (MHz)',fontsize = 8)
            ax2.set_ylabel('Average power',fontsize = 8)
            #ax2.set_ylim(min(limsq[limsq != -np.inf])/2.,max(np.log10(vis_q)))

            limsu = 10000*np.log10(vis_u)
            ax3 = fig.add_subplot(413,sharex=ax33,frameon=False)
            ax3.plot(xr,vis_u,alpha=0.6)
            ax3.set_title('Vis yx bs:%s m:%s '%(ibs , q),fontsize = 10)
            ax3.set_xlabel('Frequency (MHz)',fontsize = 8)
            ax3.set_ylabel('Average power',fontsize = 8)
            #ax3.set_ylim(min(limsu[limsu != -np.inf])/2.,max(np.log10(vis_u)))

            limsv = 1000*np.log10(vis_v)
            ax4 = fig.add_subplot(414,sharex=ax44,frameon=False)
            ax4.plot(xr,vis_v,alpha=0.6)
            ax4.set_title('Vis yy bs:%s m:%s '%(ibs , q),fontsize = 10)
            ax4.set_xlabel('Frequency (MHz)',fontsize = 8)
            ax4.set_ylabel('Average power',fontsize = 8)
            #ax4.set_ylim(min(limsv[limsv != -np.inf])/2.,max(np.log10(vis_v)))


        blue_line =mlines.Line2D([], [], color='#1f77b4', linestyle='-.',
                   label=r'$\Xi$;$\phi$ = {}$^\circ$'.format(phi[0]*15))
        orange_line =mlines.Line2D([], [], color='#ff7f0e', linestyle='-.',
                     label=r'$\Xi$;$\phi$ = {}$^\circ$'.format(phi[1]*15))
        z_line =mlines.Line2D([], [], color='k', linestyle='-',
                label='<V>')
        plt.legend(handles=[blue_line,orange_line,z_line],loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        plt.tight_layout()
        plt.savefig('/data4/paper/rkb/xiimagstorage/2457755_RFIraw/2457755.RFIraw.avgvis_{}_{}.png'.format(ibs,q))
        plt.close()
