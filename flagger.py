import numpy as np
from pyuvdata import UVData
import glob

uvd = UVData()
uvd.read_miriad('zen.2457555.12345.xx.HH.uvc')
data_dir = glob.glob("/data4/paper/HERA2015/2457555/PennData/4polminV_v2")

for file in data_dir:
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
	    flagged_data = np.ma.masked_where(f == 1, d)