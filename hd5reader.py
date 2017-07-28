import h5py
import matplotlib
from pyuvdata import UVData

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import shutil
from pylab import *
UV = UVData()

def layoftheland(data_dir):
	fn = glob.glob(''.join([data_dir, 'xi_nu_phi_vis.hdf5']))
	f = h5py.File(fn[0], 'r')
	dgrp = f["/Data"]
	for key in dgrp.keys():
 		print(key)
 	dset_nu = dgrp["nu"]
 	nu = np.asarray(dset_nu)
 	print ("Nu shape:")
 	print(nu.shape)
 	dset_phi = dgrp["phi"]
 	phi = np.asarray(dset_phi)
 	print ("Phi shape:")
 	print(phi.shape)
 	dset_tauh = dgrp["tauh"]
 	tauh = np.asarray(dset_tauh)
 	print ("Tauh shape:")
 	print(tauh.shape)
 	dset_xi = dgrp["xi"]
 	xi = np.asarray(dset_xi)
 	print ("Xi shape:")
 	print(xi.shape)


def viscalculator(data_dir):
	fn =("/home/plaplant/global_signal/Output/HERA/beam_zenith/xi_nu_phi_vis.hdf5")
	f = h5py.File(fn, 'r')
	dgrp = f["/Data"]
	dset_xi = dgrp["xi"]
	xi = np.asarray(dset_xi)
	vis_xx = xi[0, 0, 0, :]
	vis_xy = xi[0, 1, 0, :]
	vis_yx = xi[0, 2, 0, :]
	vis_yy = xi[0, 3, 0, :]
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	avg = 0
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcORR'])))
	yydatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcORR'])))
	xxdatalist2 = np.empty((56, 1024, 28), dtype=np.complex128)
	yydatalist2 = np.empty((56, 1024, 28), dtype=np.complex128)
	for miriad_file in xxdatafiles:
		UV.read_miriad(miriad_file)
		xxdatalist = np.empty((56, 1024))
		for baseline in antpairall:
			xxdata = UV.get_data(baseline)
			if xxdata.shape != (56, 1024):
				pass
			else:
				xxdatalist = np.dstack((xxdatalist, xxdata))
		if xxdatalist.shape != (56, 1024, 28):
			pass
		else:
			xxdatalist2 += xxdatalist
	for miriad_file in yydatafiles:
		UV.read_miriad(miriad_file)
		yydatalist = np.empty((56, 1024))
		for baseline in antpairall:
			yydata = UV.get_data(baseline)
			if yydata.shape != (56, 1024):
				pass
			else:
				yydatalist = np.dstack((yydatalist, yydata))
		if yydatalist.shape != (56, 1024, 28):
			pass
		else:
			yydatalist2 += yydatalist
	xxtotal = np.sum(xxdatalist2, axis=0)
	yytotal = np.sum(yydatalist2, axis=0)
	n_avg = len(xxdatafiles)*56
	xxtotalavg = xxtotal/n_avg
	yytotalavg = yytotal/n_avg
	baselineiterator = xxtotalavg[0, :]
	ax1=plt.subplot(211)
	ax1.set_title("Vis XX")
	ax1.set_ylim=(-0.25, 0.25)
	ax1.set_xlabel("Frequency (MHz)")

	ax1.plot(vis_xx, label="hd5line")
	for i, element in enumerate(baselineiterator):
			ax1.plot(xxtotalavg[:, i])
	ax2 = plt.subplot(212)
	ax2.set_title("Vis YY")
	ax2.set_ylim(-0.25, 0.25)
	ax2.set_xlabel("Frequency (MHz)")
	ax2.plot(vis_yy, label="hd5line")
	plt.legend()
	plt.tight_layout()
	fig = plt.gcf()
	fig.suptitle("Vis XX and Vis YY, HDf5 against Actual Avg Over Time")

	for i, element in enumerate(baselineiterator):
			ax2.plot(yytotalavg[:, i])
	plt.savefig("/data4/paper/rkb/viscalcgraph.png")		

def stokescreator(stokes):
	if os.path.isdir("/data4/paper/rkb/hd5saves/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/hd5saves/")
	fn = '/home/plaplant/global_signal/Output/HERA/beam_zenith/xi_nu_phi.hdf5'
	f = h5py.File(fn, 'r')
	dgrp = f["/Data"]
	dset_nu = dgrp["nu"]
	nu = np.asarray(dset_nu)
	dset_xi = dgrp["xi"]
	xi = np.asarray(dset_xi)
	if stokes == "I":
		xi_stokes = xi[0, 0, 0, :]
	elif stokes == "Q":
		xi_stokes = xi[0, 1, 0, :]
	elif stokes == "U":
		xi_stokes = xi[0, 2, 0, :]
	elif stokes == "V":
		xi_stokes = xi[0, 3, 0, :]
	plt.plot(nu, np.abs(xi_stokes), color='b', linestyle='-', label="Absolute value")
	plt.legend()
	plt.title('stokes {}'.format(stokes))
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Avg Power')
	plt.savefig('/data4/paper/rkb/hd5saves/hd5test1.png')

def baselinetest(fn):
	if os.path.isdir("/data4/paper/rkb/hd5savesgif/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/hd5savesgif/")
	f = h5py.File(fn, 'r')
	dgrp = f["/Data"]
	dset_nu = dgrp["nu"]
	nu = np.asarray(dset_nu)
	dset_xi = dgrp["xi"]
	xi = np.asarray(dset_xi)
	xi_baseline = xi[:,3, 0, 0]
	xi_angle = xi[0, 0, :, 0]
	for index2 in enumerate(np.nditer(xi_angle)):
		for index in enumerate(np.nditer(xi_baseline)):
			xi_plot = xi[index[0], 0, index2[0], : ]
			ax= plt.subplot(411)
			ax.set_title("Stokes I")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			plt.ylabel('Avg Power')
			xi_plot = xi[index[0], 1, index2[0], : ]
			ax= plt.subplot(412)
			ax.set_title("Stokes Q")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			plt.ylabel('Avg Power')
			xi_plot = xi[index[0], 2, index2[0], : ]
			ax= plt.subplot(413)
			ax.set_title("Stokes U")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			plt.ylabel('Avg Power')
			xi_plot = xi[index[0], 3, index2[0], : ]
			ax= plt.subplot(414)
			ax.set_title("Stokes V")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			plt.ylabel('Avg Power')
		plt.legend(loc="best")
		plt.xlabel('Frequency (MHz)')
		#plt.figure(figsize=(10, 15))
		plt.tight_layout()
		fig = gcf()
		fig.suptitle('tauh = {}'.format(index2[1]))
		plt.savefig('/data4/paper/rkb/hd5savesgif/hd5stokesQ{}.png'.format(index2))
		plt.clf()
	images = glob.glob('/data4/paper/rkb/hd5savesgif/*.png')
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/hd5.gif', gif,fps=1)
	shutil.rmtree('/data4/paper/rkb/hd5savesgif/')



	# compare sam's plots against .vis.uvfits file data!



# for baselin_sep in xi:

# xi = np.asarray(dset_xi)
# plt.plot(nu, np.abs(xi0))

# plt.legend()
# plt.savefig("xi.pdf")

# for key in f.keys():
# 	print(key)
# dgrp = f["/Data"]
# for key in dgrp.keys():
# 	print(key)
