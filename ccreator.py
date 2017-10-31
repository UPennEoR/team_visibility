from pyuvdata import UVData
import numpy as np
import glob
UV = UVData()

#**conjugate zac's data
#instead of stacking them and then collapsing, add as you go. Zach's data isn't aligned 
#with the actual data.

# files = [f for f in os.listdir(visibility_dir) if f[-4:] == '.uvc']
# uv = UVData()
# uv.read_miriad(files)

def ccreator(data_dir):
	zacxxdatafiles = sorted(glob.glob(''.join(['/data4/paper/HERA2015/2457458/', 'zen.*.xx.HH.uvc'])))
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvc'])))
	baselines = [(9, 20), (20, 89), (53, 31)]
	xxdatalist2 = np.empty((56, 1024, 3), dtype=np.complex128)
	UV.read_miriad(xxdatafiles[0])
	antpairall= UV.get_antpairs()
	for miriad_file in xxdatafiles:
		UV.read_miriad(miriad_file)
		xxdatalist = np.empty((56, 1024), dtype=np.complex128)
		i = 0
		for baseline in baselines:
			xxrealdata = UV.get_data(baseline)
			if i == 1:
				xxdatalist += xxrealdata
			else:
				xxdatalist = np.dstack((xxdatalist, xxrealdata))
			i += 1
		print(xxdatalist.shape)
		xxdatalist2 += xxdatalist
		print(xxdatalist2.shape)
	zacxxdatalist2 = np.empty((56, 1024, 3), dtype=np.complex128)
	for file in zacxxdatafiles:
		UV.read_miriad(file)
		zacxxdatalist = np.empty((56, 1024), dtype=np.complex128)
		i = 0
		for baseline in baselines:
			xxzacdata = UV.get_data(baseline)
			if i == 1:
				zacxxdatalist += xxzacdata
			else:
				zacxxdatalist = np.dstack((zacxxdatalist, xxzacdata))
			i += 1
		zacxxdatalist2 += zacxxdatalist
	c = []
	f = open("/data4/paper/rkb/Cvals.txt", "w")
	for i in len(baselines):
		mult = xxdatalist2[:, :, i] * zacxxdatalist2[:, :, i]
		square = xxdatalist2[:, :, i] *xxdatalist2[:, :, i]
		#now, sum it all by collapsing in time, and frequency
		xxmulttimetotal = np.sum(mult, axis = 0)
		xxmultcompletetotal = np.sum(xxtimetotal, axis = 0)
		xxsquaretimetotal = np.sum(square, axis = 0)
		xxsquarecompletetotal=np.sum(xxsquaretimetotal, axis = 0)
		#Divide
		c.append(xxmultcompletetotal/xxsquarecompletetotal)
		f.write("%s, %s \n", c[i], baseline[i])
	f.close()

