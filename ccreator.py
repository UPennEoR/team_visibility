from pyuvdata import UVData
import numpy as np
import glob
UV = UVData()

def ccreator(data_dir):
	# zacfiles = sdkfjsdhlksjhf
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvc'])))
	baselines = ['9_20', '20_89', '53_31']
	xxdatalist2 = np.empty((4032, 1024, 28), dtype=np.complex128)
	for miriad_file in xxdatafiles:
		UV.read_miriad(miriad_file)
		xxdatalist = np.empty((56, 1024))
		i = 0
		for baseline in baselines:
			xxrealdata = UV.get_data(baseline)
			if xxrealdata.shape != (56, 1024):
				pass
			else:
				xxdatalist = np.dstack((xxdatalist, xxrealdata))
				i += 1
		if xxdatalist.shape != (56, 1024, 28):
			pass
		else:
			xxdatalist2 = np.hstack(xxdatalist)
			print(xxdatalist2.shape())

	# c = []
	# f = open("/data4/paper/rkb/Cvals.txt", "w")
	# for i in len(baselines):
	# 	mult = xxdatalist2[:, :, i] * zacxxdatalist2[:, :, i]
	# 	square = xxdatalist2[:, :, i] *xxdatalist2[:, :, i]
	# 	#now, sum it all by collapsing in time, and frequency
	# 	xxmulttimetotal = np.sum(mult, axis = 0)
	# 	xxmultcompletetotal = np.sum(xxtimetotal, axis = 0)
	# 	xxsquaretimetotal = np.sum(square, axis = 0)
	# 	xxsquarecompletetotal=np.sum(xxsquaretimetotal, axis = 0)
	# 	#Divide
	# 	c.append(xxmultcompletetotal/xxsquarecompletetotal)
	# 	f.write("%s, %s" c[i], baseline[i])
	# f.close()

