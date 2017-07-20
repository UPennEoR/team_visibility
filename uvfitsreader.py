import matplotlib
matplotlib.use('Agg')
from pyuvdata import UVData
import numpy as np
import glob
import matplotlib.pyplot as plt
UV = UVData()

def uvreader(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	i = 0
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97)
		xx_data = data[:,:,0]
		xy_data = data[:,:,1]
		yx_data = data[:,:,2]
		yy_data = data[:,:,3]
		vis_xx = xx_data - yy_data
		plt.imshow((np.log10(np.abs(vis_xx))),aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		#print(xx_data.shape)
		# print(data.shape)
		# data2 = UV.get_data(UV.antnums_to_baseline(53,97))
		# print(np.all(data == data2))
		# plt.imshow(np.abs(data, data2))
		i += 1
		plt.savefig("/data4/paper/rkb/uvreadertest"+ "uvreadertest.png{}".format(i))
		plt.clf()