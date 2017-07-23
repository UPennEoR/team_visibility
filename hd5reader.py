import h5py
import numpy as np
fn = '/home/plaplant/global_signal/Output/HERA/beam_zenith/xi_nu_phi.hdf5'
f = h5py.File(fn, 'r')

for key in f.keys():
	print(key)