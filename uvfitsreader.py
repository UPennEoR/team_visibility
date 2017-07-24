import matplotlib
matplotlib.use('Agg')
from pyuvdata import UVData
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
UV = UVData()


def uvreader(data_dir):
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	i = 0
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97)
		xx_data = data[:, :, 0]
		xy_data = data[:, :, 1]
		yx_data = data[:, :, 2]
		yy_data = data[:, :, 3]
		vis_xx = xx_data - yy_data
		plt.imshow((np.log10(np.abs(vis_xx))), aspect='auto',
				   vmax=0, vmin=-6, cmap='viridis')
		# print(xx_data.shape)
		# print(data.shape)
		# data2 = UV.get_data(UV.antnums_to_baseline(53,97))
		# print(np.all(data == data2))
		# plt.imshow(np.abs(data, data2))
		i += 1
		plt.savefig("/data4/paper/rkb/uvreadertest/" +
					"uvreadertest{}.png".format(i))
		plt.clf()


def uvreader2(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreaderarraystorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreaderarraystorage/")
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	total_array = np.empty([56, 1024, 4])
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		for baseline in antpairall:
			data = UV.get_data(baseline)
			total_array = np.concatenate((total_array, data), axis=0)
			np.save("/data4/paper/rkb/uvreaderarraystorage/zenuvfitssave{}.vis.uvfits".format(baseline), total_array)
			total_array = np.empty([56, 1024, 4])
def uvreader3(data_dir):
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data('xx')
		print(data.shape)


def uvreader5(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreaderstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreaderstorage/")
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	#print (antpairall)
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		for baseline in antpairall:
			xy_data = UV.get_data(baseline[0], baseline[1], 'xy')
			data = UV.get_data(baseline)
			xy_data2 = data[:, :, 2]
			if np.array_equal(xy_data, xy_data2) == True:
				print('it is the same')
			else:
				print('not the same')
			# yy_data = data[:, :, 3]
			# vis_xx = xx_data - yy_data
			# plt.imshow((np.log10(np.abs(vis_xx))), aspect='auto',
			# 		   vmax=0, vmin=-6, cmap='viridis')
			# plt.xlabel('frequency')
			# plt.ylabel('LST')
			# plt.title("{}, {}".format(baseline, uvfits_file))
			# uvfits_file = uvfits_file.strip(data_dir)
			# plt.savefig("/data4/paper/rkb/uvreaderstorage/" +"uvreaderallantpair{},{}.png".format(baseline, uvfits_file))
			# plt.clf()

		# np.concatenate((total_array, data), axis=0)
	# print (total_array)
	#np.save("/data4/paper/rkb/zenuvfitssave.vis.uvfits", total_array)


def uvreader4(data_dir):
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97, 'xx')  # data for ant1=1, ant2=2, pol='rr'
		print(data.shape)
		times = UV.get_times(53, 97)  # times corresponding to 0th axis in data


def uvreader6(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc'])))
	for miriad_file in datafiles:
		if UV.data_array is None:
			UV.read_miriad(miriad_file)
		else:
			UV2 = UVData()
			UV2.read_miriad(miriad_file)
			UV += UV2
	return UV
