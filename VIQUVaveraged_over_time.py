from __future__ import print_function
import numpy as np
import capo
import matplotlib.pyplot as plt
import glob
from baselineorderer import get_baselines
import os
from operator import itemgetter


def avgfreqall(data_dir):
	keys = sorted(get_baselines(ex_ants=[81]))
	baselines = get_baselines(ex_ants=[81])
	my_path = '/data4/paper/rkb/'
	xx_data = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR'])))
	xy_data = sorted(glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR'])))
	yx_data = sorted(glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR'])))
	yy_data = sorted(glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR'])))
	avgstokes_dict = {}

	antstr_all = ''
	antlist = []
	for it in keys:
		x = sorted(set(baselines[it]), key=itemgetter(2))
		for elem, antstr in enumerate(x):
			ant_i = x[elem][0]
			ant_j = x[elem][1]
			slope = x[elem][2]
			antlist.append("%s_%s" % (x[elem][0], x[elem][1]))
			antstr_all += "{}_{}".format(x[elem][0], x[elem][1]) + ","
			antstr_all = antstr_all[:-1]
			print (antstr_all)
			avg_freq_i_real = None
			avg_freq_q_real = None
			avg_freq_u_real = None
			avg_freq_v_real = None

			avg_freq_i_imag = None
			avg_freq_q_imag = None
			avg_freq_u_imag = None
			avg_freq_v_imag = None
			n_avg = 0


		for i in np.arange(len(xx_data)):
				print (i,end=" ")
				#print("Reading {}...".format(xx_data[i]))
				t_xx, d_xx, f_xx = capo.miriad.read_files(xx_data, antstr=antstr_all, polstr='xx')
				#print("Reading {}...".format(xy_data[i]))
				t_xy, d_xy, f_xy = capo.miriad.read_files(xy_data, antstr=antstr_all, polstr='xy')
				#print("Reading {}...".format(yx_data[i]))
				t_yx, d_yx, f_yx = capo.miriad.read_files(yx_data, antstr=antstr_all, polstr='yx')
				#print("Reading {}...".format(yy_data[i]))
				t_yy, d_yy, f_yy = capo.miriad.read_files(yy_data, antstr=antstr_all, polstr='yy')

				print (d_xx)
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


		


				if avg_freq_i_real is None:
					avg_freq_i_real = np.zeros(len(keys), 8, 1024)
					avg_freq_i_imag = np.zeros(len(keys), 8, 1024)
					avg_freq_q_real = np.zeros(len(keys), 8, 1024)
					avg_freq_q_imag = np.zeros(len(keys), 8, 1024)
					avg_freq_u_real = np.zeros(len(keys), 8, 1024)
					avg_freq_u_imag = np.zeros(len(keys), 8, 1024)
					avg_freq_v_real = np.zeros(len(keys), 8, 1024)
					avg_freq_v_imag = np.zeros(len(keys), 8, 1024)


				for i in range(vis_xx.shape[0]):
					avgstokes_dict['%s' %(antstr)]={}
					avgstokes_dict['%s' %(antstr)]['i_real']+= stokes_I_real[i, :]
					avgstokes_dict['%s' %(antstr)]['i_imag']+= stokes_I_imag[i, :]
					avgstokes_dict['%s' %(antstr)]['q_real']+= stokes_Q_real[i, :]
					avgstokes_dict['%s' %(antstr)]['q_imag']+= stokes_Q_imag[i, :]
					avgstokes_dict['%s' %(antstr)]['u_real']+= stokes_U_real[i, :]
					avgstokes_dict['%s' %(antstr)]['u_imag']+= stokes_U_imag[i, :]
					avgstokes_dict['%s' %(antstr)]['v_real']+= stokes_V_real[i, :]
					avgstokes_dict['%s' %(antstr)]['v_imag']+= stokes_V_imag[i, :]



					n_avg += 1
				n_avg = n_avg * vis_xx.shape[0]	
				for elem, antstr in enumerate(antlist):
						   # finish averaging
					avg_freq_i_real = avgstokes_dict['%s' %(antstr)]['i_real']/n_avg
					avg_freq_i_imag = avgstokes_dict['%s' %(antstr)]['i_imag']/n_avg
					avg_freq_q_real = avgstokes_dict['%s' %(antstr)]['q_real']/n_avg
					avg_freq_q_imag = avgstokes_dict['%s' %(antstr)]['q_imag']/n_avg
					avg_freq_u_real = avgstokes_dict['%s' %(antstr)]['u_real']/n_avg
					avg_freq_u_imag = avgstokes_dict['%s' %(antstr)]['u_imag']/n_avg
					avg_freq_v_real = avgstokes_dict['%s' %(antstr)]['v_real']/n_avg
					avg_freq_v_imag = avgstokes_dict['%s' %(antstr)]['v_imag']/n_avg
					print (avg_freq_v_imag)


				np.savez(my_path+'/'+'zen.2457746.avgstokes.{}.{}.npz'.format(it,slope),
				i_real = avg_freq_i_real,
				i_imag = avg_freq_i_imag,
				q_real = avg_freq_q_real,
				q_imag = avg_freq_q_imag,
				u_real = avg_freq_u_real,
				u_imag = avg_freq_u_imag,
				v_real = avg_freq_i_real,
				v_imag = avg_freq_i_imag)

# def avgfreqcalc(data_dir, antstr):
# 	xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
# 	xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
# 	yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
# 	yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))

# 	ant_i, ant_j = map(int, antstr.split('_'))

# 	# initialize average power
# 	avg_freq = None
# 	n_avg = 0
# 	# loop over files

# 	for i in np.arange(len(xx_data)):
# 		t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx', verbose=True)
# 		t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy', verbose=True)
# 		vis_xx = d_xx[(ant_i, ant_j)]['xx']
# 		vis_yy = d_yy[(ant_i, ant_j)]['yy']
# 		channels = vis_xx.shape[1]
# 		if avg_freq is None:
# 			avg_freq = np.zeros((vis_xx.shape[1]))
# 		if stokes == "I":
# 			stokes_I = vis_xx + vis_yy
# 			for it in range(vis_xx.shape[0]):    
# 				avg_freq_i_real += (stokes_I[it, :])
# 				n_avg += 1
# 		elif stokes == "Q":
# 			stokes_Q = vis_xx - vis_yy
# 			for it in range(vis_xx.shape[0]):    
# 				avg_freq += (stokes_Q[it, :])
# 				n_avg += 1
# 	elif stokes == "V" or stokes == "U":
# 		for i in np.arange(len(xy_data)):
# 			t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy', verbose=True)
# 			t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx', verbose=True)

# 			vis_xy = d_xy[(ant_i, ant_j)]['xy']
# 			vis_yx = d_yx[(ant_i, ant_j)]['yx']
# 			channels = vis_xy.shape[1]
# 			if avg_freq is None:
# 				avg_freq = np.zeros((vis_xy.shape[1]))
# 			if stokes == "U":
# 				stokes_U = vis_xy + vis_yx
# 				for it in range(vis_xy.shape[0]):    
# 					avg_freq += (stokes_U[it, :])
# 					n_avg += 1
# 			elif stokes == "V":
# 				stokes_V = np.imag(vis_xy) - np.imag(vis_yx)
# 				for it in range(vis_yx.shape[0]):    
# 					avg_freq += np.abs(stokes_V[it, :])
# 					n_avg += 1
			

# 	# finish averaging
# 	avg_freq = np.abs(avg_freq/n_avg)
# 	return avg_freq, channels

# def avgfreqcalc2(data_dir, antstr, stokes):
# 	xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
# 	xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
# 	yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
# 	yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))

# 	ant_i, ant_j = map(int, antstr.split('_'))

# 	# initialize average power
# 	avg_freq = None
# 	n_avg = 0
# 	# loop over files
# 	for i in np.arange(len(xx_data)):
# 		t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx', verbose=True)
# 		t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy', verbose=True)
# 		t_xy, d_xy, f_xy = capo.miriad.read_files([xy_data[i]], antstr=antstr, polstr='xy', verbose=True)
# 		t_yx, d_yx, f_yx = capo.miriad.read_files([yx_data[i]], antstr=antstr, polstr='yx', verbose=True)

# 		vis_xx = d_xx[(ant_i, ant_j)]['xx']
# 		vis_yy = d_yy[(ant_i, ant_j)]['yy']
# 		vis_xy = d_xy[(ant_i, ant_j)]['xy']
# 		vis_yx = d_yx[(ant_i, ant_j)]['yx']

# 		channels = vis_xx.shape[1]
		

# 		if avg_freq is None:
# 			avg_freq = np.zeros((vis_xx.shape[1]))
# 			stokes_I = vis_xx + vis_yy
			


# 			for it in range(vis_xx.shape[0]):    
# 				avg_freq += stokes_I[it, :]
# 				n_avg += 1
# 			elif stokes == "Q":
# 				stokes_Q = vis_xx - vis_yy
# 				for it in range(vis_xx.shape[0]):    
# 					avg_freq += (stokes_Q[it, :])
# 					n_avg += 1
# 		for i in np.arange(len(xy_data)):

# 			channels = vis_xy.shape[1]
# 			if avg_freq is None:
# 				avg_freq = np.zeros((vis_xy.shape[1]))
# 			if stokes == "U":
# 				stokes_U = vis_xy + vis_yx
# 				for it in range(vis_xy.shape[0]):    
# 					avg_freq += (stokes_U[it, :])
# 					n_avg += 1
# 			elif stokes == "V":
# 				stokes_V = np.imag(vis_xy) - np.imag(vis_yx)
# 				for it in range(vis_yx.shape[0]):    
# 					avg_freq += np.abs(stokes_V[it, :])
# 					n_avg += 1
			

# 	# finish averaging
# 	avg_freq = np.abs(avg_freq/n_avg)
# 	return avg_freq, channels

# 	# plot the result
# 	# plt.plot(avg_freq)
# 	# plt.title("Average Stokes I over time")
# 	# plt.xlabel("Frequency channel")
# 	# plt.ylabel("Average power")
# 	# plt.show()
# def avgfreqloop(data_dir, stokes):
# 	baselines = ['64_88', '64_80', '9_105', '9_53', '53_104', '22_72', '20_22', '20_31', '31_96', '65_89', '10_97', '10_43', '72_105', '88_105', '22_112', '9_22', '9_64', '20_53', '53_80', '10_89', '31_89', '31_104', '43_65', '65_96', '72_112', '97_112', '22_105', '9_88', '9_20', '20_89', '43_89', '53_64', '31_53', '31_65', '80_104', '96_104']
# 	for antstr in baselines:
# 		ant_i, ant_j = map(int, antstr.split('_'))
# 		if stokes == "I":
# 			xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
# 			yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))
# 			for i in np.arange(len(xx_data)):
# 				t_xx, d_xx, f_xx = capo.miriad.read_files([xx_data[i]], antstr=antstr, polstr='xx')
# 				t_yy, d_yy, f_yy = capo.miriad.read_files([yy_data[i]], antstr=antstr, polstr='yy')
# 				vis_xx = d_xx[(ant_i, ant_j)]['xx']
# 				vis_yy = d_yy[(ant_i, ant_j)]['yy']

# 	xx_data = glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR']))
# 	#xy_data = glob.glob(''.join([data_dir, 'zen.*.xy.HH.uvcORR']))
# 	#yx_data = glob.glob(''.join([data_dir, 'zen.*.yx.HH.uvcORR']))
# 	yy_data = glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR']))

