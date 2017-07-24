import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

fn = '/home/plaplant/global_signal/Output/HERA/beam_zenith/xi_nu_phi.hdf5'
f = h5py.File(fn, 'r')

dgrp = f["/Data"]
dset_nu = dgrp["nu"]
nu = np.asarray(dset_nu)
dset_xi = dgrp["xi"]
xi = np.asarray(dset_xi)

xi_stokes = xi[0, :, 0, 0]
print (xi_stokes)

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
