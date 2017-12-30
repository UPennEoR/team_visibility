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
			yy_data = data[:, :, 1]
			xy_data = data[:, :, 2]
			yx_data = data[:, :, 3]
			
			if xx_data.shape != (56, 1024):
				pass
			else:
				xxdatalist = np.vstack((xxdatalist, xx_data))
			if yy_data.shape != (56, 1024):
				pass
			else:
				yydatalist = np.vstack((yydatalist, yy_data))
		stokesI = xxdatalist+yydatalist
		print (stokesI.shape)
		plt.imshow((np.log10(np.abs(stokesI))), aspect='auto',
					   vmax=6, vmin=-6, cmap='viridis')
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
def miriadplotter(data_dir):
	if os.path.isdir("/data4/paper/rkb/miriadplotter/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/miriadplotter/")

	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	antpairfile = datafiles[0]
	UV.read_uvfits(antpairfile)
	antpairall = (UV.get_antpairs())
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
	#collapse in time:
	xxtotal= np.sum(xxdatalist2, axis=0)
	#avg:
	n_avg = len(xxdatafiles)*56
	xxavg = xxtotal/n_avg
	baselineiterator = xxavg[0, :]
	for i, element in enumerate(baselineiterator):
		print(i)
		plt.plot(np.real(xxavg[:, i]), label="real part")
		plt.plot(np.imag(xxavg[:, i]), label="imaginary part")
		plt.legend()
		plt.ylabel("Average Power")
		plt.title("Visibility Avg Over Time, {}".format(antpairall[i-1]))
		plt.tight_layout()
		plt.savefig("/data4/paper/rkb/miriadplotter/{}.png".format(antpairall[i-1]))
		plt.clf()


def miriadtimeavgreader(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvtimeavgreaderstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvtimeavgreaderstorage/")
	# antpairall = (72,22)
	
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcOR'])))
	antpairfile = xxdatafiles[0]
	UV.read_miriad(antpairfile)
	antpairall = UV.get_antpairs()
	yydatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcOR'])))
	xxdatalist2 = np.empty((112, 1024, 28), dtype=np.complex128)
	yydatalist2 = np.empty((112, 1024, 28), dtype=np.complex128)
	for baseline in antpairall:
		xxdatalist = np.empty((112, 1024))
		for miriad_file in xxdatafiles:
			UV.read_miriad(miriad_file)
			xxdata = UV.get_data(baseline)	
			print(xxdata.shape)
			print(xxdata)
			# if xxdata.shape != (56, 1024):
			# 	pass
			# else:
			# xxdatalist = np.dstack((xxdatalist, xxdata))
		# if xxdatalist.shape != (56, 1024, 28):
		# 	pass
		# else:
		xxdatalist2 += xxdatalist
	for miriad_file in yydatafiles:
		UV.read_miriad(miriad_file)
		yydatalist = np.empty((112, 1024))
		for baseline in antpairall:
			yydata = UV.get_data(baseline)
			# if yydata.shape != (56, 1024):
			# 	pass
			# else:
			yydatalist = np.dstack((yydatalist, yydata))
		# if yydatalist.shape != (56, 1024, 28):
		# 	pass
		# else:
		yydatalist2 += yydatalist
	#collapse in time:
	xxtotal= np.sum(xxdatalist2, axis=0)
	yytotal= np.sum(yydatalist2, axis=0)
	#avg:
	n_avg = len(xxdatafiles)*112
	xxavg = xxtotal/n_avg
	yyavg = yytotal/n_avg
	print(xxavg)
	baselineiterator = xxavg
	for i in range(1, len(baselineiterator)):
		ax1 = plt.subplot(211)
		ax1.set_ylim(-0.05, 0.05)
		ax1.plot(np.real(xxavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real")
		ax2 = plt.subplot(212)
		ax2.plot(np.real(yyavg))
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary")
		plt.tight_layout()
		fig = plt.gcf()
		fig.suptitle("Visibility Avg over Time")
		plt.savefig("/data4/paper/rkb/uvtimeavgreaderstorage/{}.png".format(antpairall[i]))
		plt.clf()


def miriadtimeavgreader2(data_dir):
	if os.path.isdir("/data4/paper/rkb/miriadtimeavgreaderstorage/XX/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/miriadtimeavgreaderstorage/XX/")
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvcOR'])))
	antpairfile = xxdatafiles[0]
	UV.read_miriad(antpairfile)
	antpairall = UV.get_antpairs()
	# yydatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.yy.HH.uvcOR'])))
	for baseline in antpairall:
		xxdatalist = np.empty((112, 1024))
		for miriad_file in xxdatafiles:
			UV.read_miriad(miriad_file)
			xxdata = UV.get_data(baseline)
			if xxdata.shape != (112, 1024):
				pass
			else:
				xxdatalist = np.vstack((xxdatalist, xxdata))
		xxtotal= np.sum(xxdatalist, axis=0)
		print(xxdatalist.shape)
		n_avg = 112*xxdatalist.shape(1)
		xxavg = xxtotal/n_avg
		plt.plot(np.real(xxavg))
		plt.ylim(-1.5, 1.5)
		plt.ylabel("Average Power")
		plt.tight_layout()
		plt.title("Visibility Avg over Time, XX")
		plt.savefig("/data4/paper/rkb/miriadtimeavgreaderstorage/XX/{}.png".format(baseline))
		plt.clf()
def uvtimeavgreader(data_dir):
	if os.path.isdir("/data4/paper/rkb/uvtimeavgreaderstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/uvtimeavgreaderstorage/")
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
	#collapse in time:
	xxtotal= np.sum(xxdatalist2, axis=0)
	#avg:
	n_avg = len(xxdatafiles)*56
	xxavg = xxtotal/n_avg
	baselineiterator = xxavg[0, :]
	uvdatafiles = sorted(
		glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	uvxxdatalist = np.empty((56, 1024), dtype=np.complex128)
	uvyydatalist = np.empty((56, 1024), dtype=np.complex128)
	for i, element in enumerate(baselineiterator):
		ax1 = plt.subplot(211)
		ax1.plot(np.real(xxavg[:, i]))
		UV.read_uvfits(datafiles)
		data = UV.get_data(antpairall[i-1])
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
		uvxxtotal= np.sum(uvxxdatalist, axis=0)
		uvxxavg = uvxxtotal/n_avg
		ax1.set_ylim(-0.05, 0.05)
		ax1.plot(np.real(uvxxavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real")
		ax2 = plt.subplot(212)
		ax2.plot(np.imag(xxavg[:, i]))
		ax2.plot(np.imag(uvxxavg), 'g-', linewidth=3, label="modeldata")
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary")
		ax2.legend()
		plt.tight_layout()
		fig = plt.gcf()
		fig.suptitle("Visibility Avg over Time, {}".format(antpairall[i-1]))
	# uvdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	# for uvfits_file in datafiles:
	# 	UV.read_uvfits(uvfits_file)
	# 	data = UV.get_data(antpairall[1])
	# 	xx_data = data[:, :, 0]
	# 	yy_data = data[:, :, 1]
	# 	xy_data = data[:, :, 2]
	# 	yx_data = data[:, :, 3]
	# 	stokesI = xx_data-yy_data
		plt.savefig("/data4/paper/rkb/uvtimeavgreaderstorage/{}.png".format(antpairall[i-1]))
		plt.clf()
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


def zachtimeavgreader(data_dir):
	if os.path.isdir("/data4/paper/rkb/zachtimeavgreaderstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/zachtimeavgreaderstorage/")
	xxdatafiles = sorted(glob.glob(''.join([data_dir, '*.xx'])))
	xydatafiles = sorted(glob.glob(''.join([data_dir, '*.xy'])))
	yydatafiles = sorted(glob.glob(''.join([data_dir, '*.yy'])))
	yxdatafiles = sorted(glob.glob(''.join([data_dir, '*.yx'])))
	antpairfile = xxdatafiles[0]
	UV.read_miriad(antpairfile)
	antpairall = UV.get_antpairs()
	n_avg = len(xxdatafiles)*61
	avg = 0
	uvxxdatalist = np.empty((61, 1024), dtype=np.complex128)
	uvxydatalist = np.empty((61, 1024), dtype=np.complex128)
	uvyxdatalist = np.empty((61, 1024), dtype=np.complex128)
	uvyydatalist = np.empty((61, 1024), dtype=np.complex128)
	for file in xxdatafiles:
		UV.read_miriad(file)
	for i, element in antpairall:
		print (i)
		print('2')
		xxdata = UV.get_data(antpairall[i-1])
		print('3')
		if xxdata.shape != (61, 1024):
			pass
		else:
			uvxxdatalist += xxdata
		print('4')
		uvxxtotal= np.sum(uvxxdatalist, axis=0)
		uvxxavg = uvxxtotal/n_avg
		ax1 = plt.subplot(421)
		ax1.plot(np.real(uvxxavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real XX")
		ax2 = plt.subplot(422)
		ax2.plot(np.imag(uvxxavg), 'g-', linewidth=3, label="modeldata")
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary XX")
		ax2.legend()
		plt.tight_layout()
		for file in xydatafiles:
			UV.read_miriad(file)
			xydata = UV.get_data(antpairall[i-1])
			if xydata.shape != (61, 1024):
				pass
			else:
				uvxxdatalist += xydata
		uvxytotal= np.sum(uvxydatalist, axis=0)
		uvxyavg = uvxxtotal/n_avg
		ax1 = plt.subplot(423)
		ax1.plot(np.real(uvxyavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real XY")
		ax2 = plt.subplot(424)
		ax2.plot(np.imag(uvxyavg), 'g-', linewidth=3, label="modeldata")
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary XY")
		ax2.legend()
		plt.tight_layout()
		for file in yxdatafiles:
			UV.read_miriad(file)
			yxdata = UV.get_data(antpairall[i-1])
			if yxdata.shape != (61, 1024):
				pass
			else:
				uvyxdatalist += yxdata
		uvyxtotal= np.sum(uvyxdatalist, axis=0)
		uvyxavg = uvyxtotal/n_avg
		ax1 = plt.subplot(425)
		ax1.plot(np.real(uvyxavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real YX")
		ax2 = plt.subplot(426)
		ax2.plot(np.imag(uvyxavg), 'g-', linewidth=3, label="modeldata")
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary YX")
		ax2.legend()
		plt.tight_layout()
		for file in yydatafiles:
			UV.read_miriad(file)
			yydata = UV.get_data(antpairall[i-1])
			if yydata.shape != (61, 1024):
				pass
			else:
				uvyydatalist += yxdata
		uvyytotal= np.sum(uvyydatalist, axis=0)
		uvyyavg = uvyytotal/n_avg
		ax1 = plt.subplot(427)
		ax1.plot(np.real(uvyyavg), 'g-', linewidth=3, label="modeldata")
		ax1.set_ylabel("Average Power")
		ax1.set_title("Real YY")
		ax2 = plt.subplot(428)
		ax2.plot(np.imag(uvyyavg), 'g-', linewidth=3, label="modeldata")
		ax2.set_xlabel("Frequency (MHz)")
		ax1.set_ylabel("Average Power")
		ax2.set_title("Imaginary YY")
		ax2.legend()
		plt.tight_layout()

		fig = plt.gcf()
		fig.suptitle("Zach Model Visibility Avg over Time, {}".format(antpairall[i-1]))
			# uvdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
			# for uvfits_file in datafiles:
			# 	UV.read_uvfits(uvfits_file)
			# 	data = UV.get_data(antpairall[1])
			# 	xx_data = data[:, :, 0]
			# 	yy_data = data[:, :, 1]
			# 	xy_data = data[:, :, 2]
			# 	yx_data = data[:, :, 3]
			# 	stokesI = xx_data-yy_data
		plt.savefig("/data4/paper/rkb/zachtimeavgreaderstorage/{}.png".format(antpairall[i-1]))
		plt.clf()

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
