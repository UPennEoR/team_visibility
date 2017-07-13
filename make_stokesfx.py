import numpy as np
import capo
import matplotlib.pyplot as plt

def getstokes(data_dir):
	if os.path.isdir("/data4/paper/rkb/stokesgifstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/stokesgifstorage/")
	baselines = ['64_88', '64_80', '9_105', '9_53', '53_104', '22_72', '20_22']
	for antstr in baseline:
		xx_data = glob.glob(''.join([data_dir, 'zen.2457746.xx.HH.uvcORR']))
		xy_data = glob.glob(''.join([data_dir, 'zen.2457746.xy.HH.uvcORR']))
		yx_data = glob.glob(''.join([data_dir, 'zen.2457746.yx.HH.uvcORR']))
		yy_data = glob.glob(''.join([data_dir, 'zen.2457746.yy.HH.uvcORR']))

		ant_i, ant_j = map(int, antstr.split('_'))

		t_xx, d_xx, f_xx = capo.miriad.read_files(xx_data, antstr=antstr, polstr='xx')
		t_xy, d_xy, f_xy = capo.miriad.read_files(xy_data, antstr=antstr, polstr='xy')
		t_yx, d_yx, f_yx = capo.miriad.read_files(yx_data, antstr=antstr, polstr='yx')
		t_yy, d_yy, f_yy = capo.miriad.read_files(yy_data, antstr=antstr, polstr='yy')

		# plt.subplot(241)
		# plt.imshow(np.log10(np.abs(d_xx[(ant_i, ant_j)]['xx'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		# plt.title('xx Visibility')
		# plt.xlabel('Frequency bin')
		# plt.ylabel('LST')
		# plt.subplot(242)
		# plt.imshow(np.log10(np.abs(d_xy[(ant_i, ant_j)]['xy'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		# plt.title('xy Visibility')
		# plt.xlabel('Frequency bin')
		# plt.ylabel('LST')
		# plt.subplot(243)
		# plt.imshow(np.log10(np.abs(d_yx[(ant_i, ant_j)]['yx'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		# plt.title('yx Visibility')
		# plt.xlabel('Frequency bin')
		# plt.ylabel('LST')
		# plt.subplot(244)
		# plt.imshow(np.log10(np.abs(d_yy[(ant_i, ant_j)]['yy'])), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		# plt.title('yy Visibility')
		# plt.xlabel('Frequency bin')
		# plt.ylabel('LST')

		vis_xx = np.abs(d_xx[(ant_i, ant_j)]['xx'])**2
		vis_yy = np.abs(d_yy[(ant_i, ant_j)]['yy'])**2
		vis_yx = d_yx[(ant_i, ant_j)]['yx']
		vis_xy = d_xy[(ant_i, ant_j)]['xy']

		# print(vis_xx.shape)

		stokes_I = np.real(vis_xx + vis_yy)
		stokes_Q = np.real(vis_xx - vis_yy)
		stokes_U = np.real(vis_xy + vis_yx)
		stokes_V = np.real(1j*vis_xy - 1j*vis_yx)

		plt.subplot(245)
		plt.imshow(np.log10(stokes_I), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		plt.title('Stokes I')
		plt.xlabel('Frequency bin')
		plt.ylabel('LST')

		plt.subplot(246)
		plt.imshow(np.log10(stokes_Q), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		plt.title('Stokes Q')
		plt.xlabel('Frequency bin')
		plt.ylabel('LST')

		plt.subplot(247)
		plt.imshow(np.log10(stokes_U), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		plt.title('Stokes U')
		plt.xlabel('Frequency bin')
		plt.ylabel('LST')

		plt.subplot(248)
		plt.imshow(np.log10(stokes_V), aspect='auto', vmax=0, vmin=-6, cmap='viridis')
		plt.title('Stokes V')
		plt.xlabel('Frequency bin')
		plt.ylabel('LST')

		#plt.show()
		plt.savefig("/data4/paper/rkb/stokesgifstorage/"+ "{}.png").format(antstr)
	images = glob.glob('/data4/paper/rkb/stokesgifstorage/*.png')
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/stokesgif.gif', gif,fps=1)
	shutil.rmtree('/data4/paper/rkb/stokesgifstorage/')

