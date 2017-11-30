from pyuvdata import UVData
import numpy as np
import glob
import catalog_LST

UV = UVData()
#CONVERT ZAC'S FILES TO LST. COMBINE ALL OF ZAC'S DATA, AND SPLIT INTO 1 FILE EVERY LST HOUR OR SO
#UV.Nbls
#UV.Ntimes
#Maybe use UV.set_lsts_from_time_array() ??
#UV.get_times(baseline)
#LST = UV.lst_array
#LST_1b1 = LST(np.arange(0, UV.Nbls*UV.Ntimes, UV.Nbls))
#convert to degrees: np.degrees(LST_1b1)/15
#**conjugate zac's data
#instead of stacking them and then collapsing, add as you go. Zach's data isn't aligned 
#with the actual data.

# files = [f for f in os.listdir(visibility_dir) if f[-4:] == '.uvc']
# uv = UVData()
# uv.read_miriad(files)
#you're not actually adding via time yet. Do this. You're adding all the files together, but you need to also add the times together

#develop LST calc
#add conjugate to zac's data
#**MUST FIX: MUST MULTIPLY BEFORE SUMMINNGGGGG!!!

def zacadjuster(data_dir):
	zacxxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvc'])))
	antpairfile = zacxxatafiles[0]
	simxxdatalist = np.empty((0, 1024, 1), dtype=np.complex128) #initialize biglist
	baselines = (9, 20)
	UV.read_miriad(zacxxdatafiles)
	biglist = UV.get_data(baselines).
		#concatenate by time
	LST = UV.lst_array
	LST_1b1 = LST(np.arange(0, UV.Nbls*UV.Ntimes, UV.Nbls))
	indegrees = np.degrees(LST_1b1)/15
	for i in length(LST_1b1):

	print(LST_1b1)


def match_lst(data_dir):
	lst = UV.lst_array



def ccreator(data_dir):
	zacxxdatafiles()
	xxdatafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.xx.HH.uvc'])))
	zacxxdatafiles = sorted(glob.glob(''.join(['/data4/paper/HERA2015/2457458/', 'zen.*.xx.HH.uvc'])))

	baselines = [(9, 20), (20, 89), (53, 31)]
	xxdatalist2 = np.empty((56, 1024, 3), dtype=np.complex128)
	UV.read_miriad(xxdatafiles[0])
	antpairall= UV.get_antpairs()
	LST_1b1 = LST(np.arange(0, UV.Nbls*UV.Ntimes, UV.Nbls))
	np.degrees(LST_1b1)/15
	for miriad_file in xxdatafiles:
		UV.read_miriad(miriad_file)
		LST = UV.lst_array
		LSTstart = LST[0]
		LSTstop = LST[-1]
		output = catalog_LST.find_LST("LSTstart_LSTstop", path=".")
		file2read = output(:)
		file = zacxxdatafiles(file2read)
		for baseline in baselines:
			xxrealdata = UV.get_data(baseline)
			UV.read_miriad(file)
			zacxxdata = UV.get_data(baseline)
			while j <= 1024:
				LST[j] = LSTstartindex
				LST[j+1]= LSTstopindex
				output2 = catalog_LST.find_LST("LSTstartindex_LSTstopindex", path=".")
				index = output2[2]
				zacindexeddata = zacxxdata(:, index)
				j += 1


	



	for miriad_file in xxdatafiles:
		xxdatalist = np.empty((56, 1024), dtype=np.complex128)
		i = 0
		for baseline in baselines:
			xxrealdata = UV.get_data(baseline)
			UV.get_times(baseline)

			if i == 0:
				xxdatalist += xxrealdata
			else:
				xxdatalist = np.dstack((xxdatalist, xxrealdata))
			print(i)
			i += 1
		print(xxdatalist.shape)
		xxdatalist = np.concatenate(xxdatalist, axis = 0)
		xxdatalist2 += xxdatalist
		print(xxdatalist2.shape)
	zacxxdatalist2 = np.empty((56, 1024, 3), dtype=np.complex128)
	for file in zacxxdatafiles:
		UV.read_miriad(file)
		zacxxdatalist = np.empty((56, 1024), dtype=np.complex128)
		i = 0
		for baseline in baselines:
			xxzacdata = UV.get_data(baseline)
			if i == 0:
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

