import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import shutil
from pylab import *



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
			xi_plot = xi[index[0], 1, index2[0], : ]
			ax= plt.subplot(412)
			ax.set_title("Stokes Q")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			xi_plot = xi[index[0], 2, index2[0], : ]
			ax= plt.subplot(413)
			ax.set_title("Stokes U")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
			xi_plot = xi[index[0], 3, index2[0], : ]
			ax= plt.subplot(414)
			ax.set_title("Stokes V")
			ax.plot(nu, np.abs(xi_plot), linestyle='-', label="{}".format(index[1]))
			ax.set_ylim([0, 0.00015])
		plt.tight_layout()
		plt.legend(loc="best")
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Avg Power')
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
