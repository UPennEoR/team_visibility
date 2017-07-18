#baseline pairs:
#72_112, complex.conjugate(97_112), complex.conj(22_105), complex.conj(9_88), 9_20, 20_89, complex.conj(43_89), complex.conj(53_64), comlpex.conj(53_31), complex.conj(31_65), 80_104, complex.conjugate(96_104), complex.conj(9_88)

#complex.conjugate(72_112), 97_112, 22_105, 9_88, complex.conj(9_20), complex.conj(89_20), 43_89, 53_64, 31_53, complex.conj(65_31), complex.conj(80_104), 96_10

#72_105, complex.conj(88_105), complex.conj(22_112), complex.conj(9_22), 9_64, 20_54, 53_80, 10_89, complex.conj(31_89), 31_104, 43_65, 65_96

#88_105, complex.conj(72_105), 22_112, 9_22, complex.conj(9_64), complex.conj(20_54), complex.conj(53_80), complex.conj(10_89), 31_89, complex.conj(31_104), complex.conj(43_65), complex.conj(65_96)

#complex.conj(64_88), 64_80, complex.conj(9_105), 9_53, 53_104, complex.conj(22_72), complex.conj(20_22), 20_31, 31_96, complex.conj(65_89), complex.conj(10_97), 10_43

#64_88, complex.conj(64_80), 9_105, complex.conj(9_53), complex.conj(53_104), 22_72, 20_22, complex.conj(20_31), complex.conj(31_96), 65_89, 10_97, complex.conj(10_43)

#72_97, complex.conj(10_22), complex.conj(20_88), 9_89, 20_43, complex.conj(31_64), 53_65, 80_96

#complex.conj(72_97), 10_22, 20_88, complex.conj(9_89), complex.conj(20_43), 31_64, complex.conj(53_65), complex.conj(80_96)

#72_88, complex.conj(9_112), 22_64, complex.conj(20_97), 10_31, 89_104, 43_96

#complex.conj(72_88), 9_112, complex.conj(22_64), 20_97, complex.conj(10_31), complex.conj(89_104), complex.conj(43_96)

#complex.conj(80_88), complex.conj(53_105), 9_104, complex.conj(20_72), 22_31, 20_96, complex.conj(89_112), complex.conj(43_97)

#80_88, 53_105, complex.conj(9_104), 20_72, complex.conj(22_31), complex.conj(20_96), 89_112, 43_97


import matplotlib
matplotlib.use('Agg')
import numpy as np
import capo
import matplotlib.pyplot as plt
import os
import shutil
import glob
import imageio
from VIQUVaveraged_over_time import avgfreqcalc2

def getstokes(data_dir):
	xx_data = glob.glob(''.join([data_dir, 'zen.2457746.38268.xx.HH.uvcORR']))
	xy_data = glob.glob(''.join([data_dir, 'zen.2457746.38268.xy.HH.uvcORR']))
	yx_data = glob.glob(''.join([data_dir, 'zen.2457746.38268.yx.HH.uvcORR']))
	yy_data = glob.glob(''.join([data_dir, 'zen.2457746.38268.yy.HH.uvcORR']))

	if os.path.isdir("/data4/paper/rkb/stokesgifstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/stokesgifstorage/")
	#baseline = ['72_112', '9_20', '20_89', '80_104']
	#
	for i in np.arange(len(baseline)):
		baseline_1 = baseline[i]
		for j in np.arange(len(baseline)):
			baseline_2 = baseline[j]
			if baseline_1 != baseline_2:
				ant_i1, ant_j1 = map(int, baseline_1.split('_'))
				ant_i2, ant_j2 = map(int, baseline_2.split('_'))


				t_xx1, d_xx1, f_xx1 = capo.miriad.read_files(xx_data, antstr=baseline_1, polstr='xx')
				t_xy1, d_xy1, f_xy1 = capo.miriad.read_files(xy_data, antstr=baseline_1, polstr='xy')
				t_yx1, d_yx1, f_yx1 = capo.miriad.read_files(yx_data, antstr=baseline_1, polstr='yx')
				t_yy1, d_yy1, f_yy1 = capo.miriad.read_files(yy_data, antstr=baseline_1, polstr='yy')


				t_xx2, d_xx2, f_xx2 = capo.miriad.read_files(xx_data, antstr=baseline_2, polstr='xx')
				t_xy2, d_xy2, f_xy2 = capo.miriad.read_files(xy_data, antstr=baseline_2, polstr='xy')
				t_yx2, d_yx2, f_yx2 = capo.miriad.read_files(yx_data, antstr=baseline_2, polstr='yx')
				t_yy2, d_yy2, f_yy2 = capo.miriad.read_files(yy_data, antstr=baseline_2, polstr='yy')
				plt.subplot(141)
				plt.imshow(np.log10(np.abs(d_xx1[(ant_i1, ant_j1)]['xx']))-np.log10(np.abs(d_xx2[(ant_i2, ant_j2)]['xx'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('xx Visibility')
				plt.xlabel('Frequency bin')
				plt.ylabel('LST')
				plt.subplot(142)
				plt.imshow(np.log10(np.abs(d_xy1[(ant_i1, ant_j1)]['xy']))-np.log10(np.abs(d_xy2[(ant_i2, ant_j2)]['xy'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('xy Visibility')
				plt.xlabel('Frequency bin')
				plt.subplot(143)
				plt.imshow(np.log10(np.abs(d_yx1[(ant_i1, ant_j1)]['yx']))-np.log10(np.abs(d_yx2[(ant_i2, ant_j2)]['yx'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('yx Visibility')
				plt.xlabel('Frequency bin')
				plt.subplot(144)
				plt.imshow(np.log10(np.abs(d_yy1[(ant_i1, ant_j1)]['yy']))-np.log10(np.abs(d_yy2[(ant_i2, ant_j2)]['yy'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('yy Visibility')
				plt.xlabel('Frequency bin')
				plt.colorbar()
				plt.suptitle('{0} - {1}'.format(baseline_1, baseline_2))

				# vis_xx = np.abs(d_xx[(ant_i, ant_j)]['xx'])**2
				# vis_yy = np.abs(d_yy[(ant_i, ant_j)]['yy'])**2
				# vis_yx = d_yx[(ant_i, ant_j)]['yx']
				# vis_xy = d_xy[(ant_i, ant_j)]['xy']

				# # print(vis_xx.shape)

				# stokes_I = np.real(vis_xx + vis_yy)
				# stokes_Q = np.real(vis_xx - vis_yy)
				# stokes_U = np.real(vis_xy + vis_yx)
				# stokes_V = np.real(1j*vis_xy - 1j*vis_yx)
				# plt.tight_layout()
				# plt.subplot(245)
				# plt.imshow(np.log10(stokes_I), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes I')
				# plt.xlabel('Frequency bin')
				# plt.ylabel('LST')

				# plt.subplot(246)
				# plt.imshow(np.log10(stokes_Q), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes Q')
				# plt.xlabel('Frequency bin')

				# plt.subplot(247)
				# plt.imshow(np.log10(stokes_U), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes U')
				# plt.xlabel('Frequency bin')

				# plt.subplot(248)
				# plt.imshow(np.log10(stokes_V), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes V')
				# plt.xlabel('Frequency bin')
				# plt.colorbar()

				#plt.show()
				plt.savefig("/data4/paper/rkb/stokesgifstorage/"+ '{}, {}; time is {}.png'.format(baseline_1, baseline_2))
				plt.clf()
	images = glob.glob('/data4/paper/rkb/stokesgifstorage/*.png')
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/stokesgif.gif', gif,fps=1)
	shutil.rmtree('/data4/paper/rkb/stokesgifstorage/')


	def getstokes2(data_dir):
	if os.path.isdir("/data4/paper/rkb/stokesgifstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/stokesgifstorage/")
	#baseline = ['72_112', '9_20', '20_89', '80_104']
	#
	for i in np.arange(len(baseline)):
		baseline_1 = baseline[i]
		for j in np.arange(len(baseline)):
			baseline_2 = baseline[j]
			if baseline_1 != baseline_2:
				ant_i1, ant_j1 = map(int, baseline_1.split('_'))
				ant_i2, ant_j2 = map(int, baseline_2.split('_'))


				t_xx1, d_xx1, f_xx1 = capo.miriad.read_files(xx_data, antstr=baseline_1, polstr='xx')
				t_xy1, d_xy1, f_xy1 = capo.miriad.read_files(xy_data, antstr=baseline_1, polstr='xy')
				t_yx1, d_yx1, f_yx1 = capo.miriad.read_files(yx_data, antstr=baseline_1, polstr='yx')
				t_yy1, d_yy1, f_yy1 = capo.miriad.read_files(yy_data, antstr=baseline_1, polstr='yy')


				t_xx2, d_xx2, f_xx2 = capo.miriad.read_files(xx_data, antstr=baseline_2, polstr='xx')
				t_xy2, d_xy2, f_xy2 = capo.miriad.read_files(xy_data, antstr=baseline_2, polstr='xy')
				t_yx2, d_yx2, f_yx2 = capo.miriad.read_files(yx_data, antstr=baseline_2, polstr='yx')
				t_yy2, d_yy2, f_yy2 = capo.miriad.read_files(yy_data, antstr=baseline_2, polstr='yy')

				avg_freq1, channels = avgfreqcalc2(data_dir, baseline1)


				plt.imshow(np.log10(np.abs(d_xx1[(ant_i1, ant_j1)]['xx']))-np.log10(np.abs(d_xx2[(ant_i2, ant_j2)]['xx'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('xx Visibility')
				plt.xlabel('Frequency bin')
				plt.ylabel('LST')
				plt.subplot(142)
				plt.imshow(np.log10(np.abs(d_xy1[(ant_i1, ant_j1)]['xy']))-np.log10(np.abs(d_xy2[(ant_i2, ant_j2)]['xy'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('xy Visibility')
				plt.xlabel('Frequency bin')
				plt.subplot(143)
				plt.imshow(np.log10(np.abs(d_yx1[(ant_i1, ant_j1)]['yx']))-np.log10(np.abs(d_yx2[(ant_i2, ant_j2)]['yx'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('yx Visibility')
				plt.xlabel('Frequency bin')
				plt.subplot(144)
				plt.imshow(np.log10(np.abs(d_yy1[(ant_i1, ant_j1)]['yy']))-np.log10(np.abs(d_yy2[(ant_i2, ant_j2)]['yy'])), aspect='auto', vmax=5, vmin=-6, cmap='viridis')
				plt.title('yy Visibility')
				plt.xlabel('Frequency bin')
				plt.colorbar()
				plt.suptitle('{0} - {1}'.format(baseline_1, baseline_2))

				# vis_xx = np.abs(d_xx[(ant_i, ant_j)]['xx'])**2
				# vis_yy = np.abs(d_yy[(ant_i, ant_j)]['yy'])**2
				# vis_yx = d_yx[(ant_i, ant_j)]['yx']
				# vis_xy = d_xy[(ant_i, ant_j)]['xy']

				# # print(vis_xx.shape)

				# stokes_I = np.real(vis_xx + vis_yy)
				# stokes_Q = np.real(vis_xx - vis_yy)
				# stokes_U = np.real(vis_xy + vis_yx)
				# stokes_V = np.real(1j*vis_xy - 1j*vis_yx)
				# plt.tight_layout()
				# plt.subplot(245)
				# plt.imshow(np.log10(stokes_I), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes I')
				# plt.xlabel('Frequency bin')
				# plt.ylabel('LST')

				# plt.subplot(246)
				# plt.imshow(np.log10(stokes_Q), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes Q')
				# plt.xlabel('Frequency bin')

				# plt.subplot(247)
				# plt.imshow(np.log10(stokes_U), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes U')
				# plt.xlabel('Frequency bin')

				# plt.subplot(248)
				# plt.imshow(np.log10(stokes_V), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
				# plt.title('Stokes V')
				# plt.xlabel('Frequency bin')
				# plt.colorbar()

				#plt.show()
				plt.savefig("/data4/paper/rkb/stokesgifstorage/"+ '{}, {}; time is {}.png'.format(baseline_1, baseline_2))
				plt.clf()
	images = glob.glob('/data4/paper/rkb/stokesgifstorage/*.png')
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/stokesgif.gif', gif,fps=1)
	shutil.rmtree('/data4/paper/rkb/stokesgifstorage/')



