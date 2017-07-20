from pyuvdata import UVData
import numpy as np
import glob
UV = UVData()

def uvreader(data_dir):
	datafiles = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	for uvfits_file in datafiles:
		UV.read_uvfits(uvfits_file)
		data = UV.get_data(53, 97)
		print(data.shape)
		data2 = UV.get_data(UV.antnums_to_baseline(53,97))
		print(np.all(data == data2))
		plt.imshow(np.abs(data, data2))