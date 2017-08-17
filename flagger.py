import numpy as np
from pyuvdata import UVData
import glob
uvd = UVData()


data_dir = "/data4/paper/HERA2015/2457555/PennData/RFI_flag2/"
datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvcOR'])))

for file in datafiles:
	print(file)
	uvd.read_miriad(file)

# iterate over all baselines
# "key" is the baseline pair, saved as (ant1, ant2, pol)
#   for instance, (89, 96, 'XX')
# "d" is the data for that baseline-pair, and is of
#   size (Ntime, Nfreq) like you're used to
	for key, d in uvd.antpairpol_iter():
	    ind1, ind2, ipol = uvd._key2inds(key)

	    # get associated flags
	    f = uvd.flag_array[ind1, 0, :, ipol]

	    # apply them to the data
	    print (d.shape)
	    flagged_data = np.ma.masked_where(f == 0.5, d)