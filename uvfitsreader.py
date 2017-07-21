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
		plt.savefig("/data4/paper/rkb/uvreadertest/"+ "uvreadertest{}.png".format(i))
		plt.clf()

def uvreader2(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	total_array = np.empty([56, 1024, 4])
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97)
		print (data)
		np.concatenate((total_array, data), axis=0)
	#print (total_array)
	#np.save("/data4/paper/rkb/zenuvfitssave.vis.uvfits", total_array)

def uvreader3(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data('xx')
		print(data.shape)

def uvreader5(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	total_array = np.empty([56, 1024, 4])
	#allpairs = UV.get_antpairs()
	print(UV.get_ants())  # All unique antennas in data
	print(UV.get_baseline_nums())  # All baseline nums in data
	print(UV.get_antpairs())  # All (ordered) antenna pairs in data (same info as baseline_nums)
	print(UV.get_antpairpols)  # All antenna pairs and polariations.

	# for uvfits_file in datafiles:
	# 	UV.read_uvfits(uvfits_file)
	# 	for baseline in allpairs:
	# 		data = UV.get_data(baseline)
	# 		print (data)
		#np.concatenate((total_array, data), axis=0)
	#print (total_array)
	#np.save("/data4/paper/rkb/zenuvfitssave.vis.uvfits", total_array)



def uvreader4(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97, 'xx')  # data for ant1=1, ant2=2, pol='rr'
		print(data.shape)
		times = UV.get_times(53, 97)  # times corresponding to 0th axis in data

