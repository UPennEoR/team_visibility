import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio

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
	if os.path.isdir("/data4/paper/rkb/hd5saves/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/hd5saves/")
	f = h5py.File(fn, 'r')
	dgrp = f["/Data"]
	dset_nu = dgrp["nu"]
	nu = np.asarray(dset_nu)
	dset_xi = dgrp["xi"]
	xi = np.asarray(dset_xi)
	xi_baseline = xi[:, 0, 0, 0]
	for index in (len(np.nditer(xi_baseline))-1):
		xi_plot = xi[index, 0, 0, 0, : ]
		plt.plot(nu, np.abs(xi_stokes), linestyle='-', label="{}".format(index))
	plt.legend()
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Avg Power')
	plt.savefig('/data4/paper/rkb/hd5saves/hd5test2.png')

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
