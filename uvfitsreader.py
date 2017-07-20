from pyuvdata import UVData
import numpy as np
import glob
UV = UVData()

def uvreader(data_dir):
	uvfits_file = sorted(glob.glob(''.join([data_dir, 'zen.*.HH.uvc.vis.uvfits'])))
	UV.read_uvfits(uvfits_file)
	#UV.write_miriad('day2_TDEM0003_10s_norx_1src_1spw.uv')  # write out the miriad file
	print(UV.get_ants())  # All unique antennas in data
	print(UV.get_baseline_nums())  # All baseline nums in data
	print(UV.get_antpairs())  # All (ordered) antenna pairs in data (same info as baseline_nums)
	print(UV.get_antpairpols)  # All antenna pairs and polariations.

