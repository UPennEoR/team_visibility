import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


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
xi_stokesI = xi[0, 1, 0, :]
plt.plot(nu, xi_stokesI.real, color='b', linestyle='-', label="I real")
plt.plot(nu, xi_stokesI.imag, color='b', linestyle='-', label="I real")
plt.legend()
plt.savefig('/data4/paper/rkb/hd5saves/hd5test1.png')

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
