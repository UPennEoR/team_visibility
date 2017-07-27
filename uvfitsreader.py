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
def uvantpairgetter(data_dir):
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	print (antpairall)

def uvwaterfallstacker(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreaderwaterfallstacker/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreaderwaterfallstacker/")

	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	for baseline in antpairall:
		avg = 0
		xxdatalist = np.empty((56, 1024), dtype=np.complex128)
		yydatalist = np.empty((56, 1024), dtype=np.complex128)
		for uvfits_file in datafiles:
			UV.read_uvfits(uvfits_file)
			data = UV.get_data(baseline)
			xx_data = data[:, :, 0]
			print (xx_data.shape)
			yy_data = data[:, :, 1]
			xy_data = data[:, :, 2]
			yx_data = data[:, :, 3]
			
			if xx_data.shape != (56, 1024, 28):
				pass
			else:
				xxdatalist = np.hstack((xxdatalist, xx_data))
			if yy_data.shape != (56, 1024, 28):
				pass
			else:
				yydatalist = np.hstack((yydatalist, yy_data))
		stokesI = xx_data+yy_data
		plt.imshow((np.log10(np.abs(stokesI))), aspect='auto',
					   vmax=0, vmin=-6, cmap='viridis')
		plt.xlabel('frequency')
		plt.ylabel('LST')
		plt.colorbar()
		plt.title("{}, {} Stokes I".format(baseline, uvfits_file))
		uvfits_file = uvfits_file.strip(data_dir)
		plt.savefig("/data4/paper/rkb/uvreaderwaterfallstacker/" +"uvreaderallantpair{},{}.png".format(baseline, uvfits_file))
		plt.clf()
def uvwaterfallreader(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreaderwaterfallstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreaderwaterfallstorage/")
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()

	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		for baseline in antpairall:
			data = UV.get_data(baseline)
			xx_data = data[:, :, 0]
			yy_data = data[:, :, 1]
			xy_data = data[:, :, 2]
			yx_data = data[:, :, 3]
			stokesI = xx_data + yy_data
			plt.imshow((np.log10(np.abs(stokesI))), aspect='auto',
					   vmax=0, vmin=-6, cmap='viridis')
			plt.xlabel('frequency')
			plt.ylabel('LST')
			plt.colorbar()
			plt.title("{}, {} Stokes I".format(baseline, uvfits_file))
			uvfits_file = uvfits_file.strip(data_dir)
			plt.savefig("/data4/paper/rkb/uvreaderwaterfallstorage/" +"uvreaderallantpair{},{}.png".format(baseline, uvfits_file))
			plt.clf()

def uvtimeavgreader(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreaderstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreaderstorage/")
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
	stokesI = xxdatalist2+yydatalist2
	#collapse in time:
	stokesItotal= np.sum(stokesI, axis=0)
	#avg:
	n_avg = len(xxdatafiles)*56
	stokesIavg = stokesItotal/n_avg
	averager = stokesIavg[0, :]
	ax1 = plt.subplot(211)
	ax1.set_ylim(-0.25, 0.25)
	for i, element in enumerate(averager):
			ax1.plot(np.real(stokesIavg[:, i]))
	uvdatafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	uvxxdatalist = np.empty((56, 1024), dtype=np.complex128)
	uvyydatalist = np.empty((56, 1024), dtype=np.complex128)
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(antpairall[1])
		xx_data = data[:, :, 0]
		yy_data = data[:, :, 1]
		xy_data = data[:, :, 2]
		yx_data = data[:, :, 3]
		if xxdata.shape != (56, 1024, 28):
			pass
		else:
			uvxxdatalist += xx_data
		if yydata.shape != (56, 1024, 28):
			pass
		else:
			uvyydatalist += yy_data
	uvstokesI = uvxxdatalist+uvyydatalist
	uvstokesItotal= np.sum(uvstokesI, axis=0)
	uvstokesIavg = uvstokesItotal/n_avg
	ax1.plot(np.real(uvstokesIavg), 'g-', linewidth=3, label="modeldata")
	ax1.set_ylabel("Average Power")
	ax1.set_title("Real")
	ax2 = plt.subplot(212)
	for i, element in enumerate(averager):
			ax2.plot(np.imag(stokesIavg[:, i]))
	ax2.plot(np.imag(uvstokesIavg), 'g-', linewidth=3, label="modeldata")
	ax2.set_xlabel("Frequency (MHz)")
	ax1.set_ylabel("Average Power")
	ax2.set_title("Imaginary")
	ax2.legend()
	plt.tight_layout()
	fig = plt.gcf()
	fig.suptitle("Stokes I Avg over Time")
	# uvdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	# for uvfits_file in datafiles:
	# 	UV.read_uvfits(uvfits_file)
	# 	data = UV.get_data(antpairall[1])
	# 	xx_data = data[:, :, 0]
	# 	yy_data = data[:, :, 1]
	# 	xy_data = data[:, :, 2]
	# 	yx_data = data[:, :, 3]
	# 	stokesI = xx_data-yy_data
	plt.savefig("/data4/paper/rkb/testtesttest.png")
	# plt.title('ActualUV Avged Over Time {} {}'.format(baseline, miriad_file))
	# for uvfits_file in datafiles:
	# 	UV.read_uvfits(uvfits_file)
	# 	for baseline in antpairall:
	# 		data = UV.get_data(baseline)
	# 		xx_data = data[:, :, 0]
	# 		yy_data = data[:, :, 1]
	# 		xy_data = data[:, :, 2]
	# 		yx_data = data[:, :, 3]
	# 		stokesI = xx_data-yy_data
	# 		averager = stokesI[:,0]
	# 		for index, element in enumerate(np.nditer(averager[0])):
	# 			avg += stokesI[:,index]
	# 		n_avg = avg/len(np.nditer(averager))
	# 		plt.plot(n_avg)
	# 		plt.xlabel('frequency')
	# 		plt.ylabel('avg power')
	# 		uvfits_file = uvfits_file.strip(data_dir)
	# 		plt.title('UV Avged over Time {} {}'.format(baseline, uvfits_file))
	# 		plt.savefig("/data4/paper/rkb/uvreaderstorage/modelvisavgedtime{}{}.png".format(baseline, uvfits_file))
	# 		plt.clf()
#pull out vis. THere shouldn't be any variation between identical baselines. Average over time. THen, overplot with the avged over time uvc files 

		# np.concatenate((total_array, data), axis=0)
	# print (total_array)
	#np.save("/data4/paper/rkb/zenuvfitssave.vis.uvfits", total_array)
def uvtimeavgreader2(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvreader2storage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvreader2storage/")
	datafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = UV.get_antpairs()
	for baseline in antpairall:
		avg = 0
		xxdatalist = np.empty((56, 1024), dtype=np.complex128)
		yydatalist = np.empty((56, 1024), dtype=np.complex128)
		for uvfits_file in datafiles:
			UV.read_uvfits(uvfits_file)
			data = UV.get_data(baseline)
			xx_data = data[:, :, 0]
			yy_data = data[:, :, 1]
			xy_data = data[:, :, 2]
			yx_data = data[:, :, 3]
			
			if xx_data.shape != (56, 1024, 28):
				pass
			else:
				xxdatalist += xx_data
			if yy_data.shape != (56, 1024, 28):
				pass
			else:
				yydatalist += yy_data
		stokesI = xx_data+yy_data
		print (stokesI.shape)
		stokesItotal= np.sum(stokesI, axis=0)
		print (stokesItotal.shape)
		for index, element in enumerate(np.nditer(stokesItotal[0])):
			avg += stokesI[:,index]
		n_avg = avg/len(np.nditer(stokesItotal))
		print (n_avg.shape)
		plt.plot(np.real(n_avg), 'g-', label="imaginary")
		plt.plot(np.imag(n_avg), label="real")
		plt.legend()
		plt.xlabel('frequency')
		plt.ylabel('avg power')
		uvfits_file = uvfits_file.strip(data_dir)
		plt.title('Model UV Avged over Time {} {}'.format(baseline, uvfits_file))
		plt.savefig("/data4/paper/rkb/uvreader2storage/modelvisavgedtime{}{}.png".format(baseline, uvfits_file))
		plt.clf()

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
