import capo
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
from VIQUVaveraged_over_time import avgfreqcalc


#7/5/17 at 23:51; running into error "UnboundLocalError: local variable 'uv' referenced before assignment"


def delaytransform(data_dir):
	files= glob.glob(''.join([data_dir, 'zen.2457746.*.*.HH.uvcORR']))
	t, d, f = capo.miriad.read_files(files, antstr='cross', polstr='xx', verbose=True)
	
	d_short = d[(72,97)]['xx']
	d_long = d[(43,88)]['xx']

	#fourier transform
	d_fft_short=np.fft.fftshift(np.fft.ifft(d_short, axis=1), axes=1)
	d_fft_long=np.fft.fftshift(np.fft.ifft(d_long, axis=1), axes=1)
	delays = np.fft.fftshift(np.fft.fftfreq(d_fft_short.shape[1], .1/d_fft_short.shape[1])) # fftfreq takes in (nchan, chan_spacing)
	#convert chan_spacing from GHz to ns (ask paul about this for clarification!!)
	d_start = delays[0]
	d_end = delays[-1]
	t_start = d_short.shape[0]

	plt.subplot(121)
	plt.imshow(np.log10(np.abs(d_fft_short)), aspect='auto', cmap='jet', vmax=0, vmin = -4, extent=[d_start, d_end, t_start,0])
	plt.title('short: 72_97')
	plt.ylabel('Time')
	plt.xlabel('Delay [ns]')
	plt.tight_layout()

	plt.subplot(122)
	plt.imshow(np.log10(np.abs(d_fft_long)), aspect='auto', cmap='jet', vmax=0, vmin = -4, extent=[d_start, d_end, t_start,0])
	plt.title('long: 88_43')
	plt.ylabel('Time')
	plt.xlabel('Delay [ns]')
	plt.tight_layout()
	plt.savefig('/data4/paper/rkb/delay.png')

def delaytransformlooped(data_dir):
	#long baselines defined by 2 or greater (abc) vector jumps
	long_baselines = [(43, 88), (72,96), (80,97)]
	short_baselines = [(9,89), (9, 112), (10,22)]
	files= glob.glob(''.join([data_dir, 'zen.2457746.*.*.HH.uvcORR']))
	counter = min(len(long_baselines), len(short_baselines))
	counter2 = np.arange(counter)
	t, d, f = capo.miriad.read_files(files, antstr='cross', polstr='xx', verbose=True)
	i = 0
	while i <=(counter-1):
		#files = glob.glob(''.join([data_dir, file]))
		d_short = d[short_baselines[i]]['xx']
		d_long = d[long_baselines[i]]['xx']

		#fourier transform
		d_fft_short=np.fft.fftshift(np.fft.ifft(d_short, axis=1), axes=1)
		d_fft_long=np.fft.fftshift(np.fft.ifft(d_long, axis=1), axes=1)
		delays = np.fft.fftshift(np.fft.fftfreq(d_fft_short.shape[1], .1/d_fft_short.shape[1])) # fftfreq takes in (nchan, chan_spacing)
		#convert chan_spacing from GHz to ns (ask paul about this for clarification!!)
		d_start = delays[0]
		d_end = delays[-1]
		t_start = d_short.shape[0]

		plt.subplot(121)	
		plt.imshow(np.log10(np.abs(d_fft_short)), aspect='auto', cmap='jet', vmax=0, vmin = -4, extent=[d_start, d_end, t_start,0])
		plt.title('short: {}'.format(short_baselines[i]))
		plt.ylabel('Time')
		plt.xlabel('Delay [ns]')
		plt.tight_layout()

		plt.subplot(122)	
		plt.imshow(np.log10(np.abs(d_fft_long)), aspect='auto', cmap='jet', vmax=0, vmin = -4, extent=[d_start, d_end, t_start,0])
		plt.title('long: {}'.format(long_baselines[i]))
		plt.ylabel('Time')
		plt.xlabel('Delay [ns]')
		plt.tight_layout()
		plt.savefig("/data4/paper/rkb/gifstorage/"+str(counter2[i])+".png")
		# filename = ("/Users/rbenefo/Desktop/HERAwork/DATADUMP/gifstorage/"+str(counter2[i])+".png")
		i +=1
		# for filez in filename:
		# 	gif.append()

	#convert output to gif form
	
	images = glob.glob('/data4/paper/rkb/gifstorage/*.png')
	gif = []
	for filename in images:
   		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/gifstorage/delaygif.gif', gif,fps=3)
def delaytransformv1(data_dir):
	antstr = "10_72"
	ant_i, ant_j = map(int, antstr.split('_'))
	d_transform = np.fft.ifft(avgfreqcalc(data_dir))
	d_transform = (np.fft.fftshift(d_transform))
	d_transform = np.abs(d_transform)
	plt.plot(np.log10((d_transform)))
	plt.xlabel('Delay(ns)')
	plt.xlim(350, 650)
	plt.ylabel('Time')
	plt.title('XX Delay Transform'+antstr)
	plt.savefig("/data4/paper/rkb/"+'{}.png'.format(antstr))




