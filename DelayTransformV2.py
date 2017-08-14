import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import capo
import numpy as np
import glob
import imageio
import shutil
import os
import aipy
from VIQUVaveraged_over_time import avgfreqcalc
import hsa7458_v001 as cal

def calculate_baseline(pair):
	antennae = cal.prms['antpos_ideal']
	dx = antennae[pair[0]]['top_x'] - antennae[pair[1]]['top_x']
	dy = antennae[pair[0]]['top_y'] - antennae[pair[1]]['top_y']
	baseline = np.around([np.sqrt(dx**2. + dy**2.)],3)[0] #XXX this may need tuning
	return baseline

def get_baselines(ex_ants=[]):
	calfile = open("/data4/paper/rkb/hsa7458_v001.py")
	try:
		print 'reading, %s'%calfile
		exec("import {cfile} as cal".format(cfile=calfile))
		antennae = cal.prms['antpos_ideal']
	except ImportError:
		raise Exception("Unable to import {cfile}.".format(cfile=calfile))
	calfile.close()
	baselines = {}
	if antenna_i == antenna_j:
		pass
	elif antenna_i < antenna_j:
		pair = (antenna_i, antenna_j)
	baseline = calculate_baseline(antennae, pair)

	if (baseline not in baselines):
		baselines[baseline] = [pair]
	elif (pair in baselines[baseline]):
		pass
	else:
		baselines[baseline].append(pair)
	return baselines


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
	plt.imshow(np.log10(np.abs(d_fft_short)), aspect='auto', cmap='jet', vmax=0, vmin = -6, extent=[d_start, d_end, t_start,0])
	plt.xlim(-250, 250)
	plt.title('short: 72_97')
	plt.ylabel('Time')
	plt.xlabel('Delay [ns]')
	plt.tight_layout()

	plt.subplot(122)
	plt.imshow(np.log10(np.abs(d_fft_long)), aspect='auto', cmap='jet', vmax=0, vmin = -4, extent=[d_start, d_end, t_start,0])
	plt.xlim(-250, 250)
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
		plt.savefig("/data4/paper/rkb/gifstorage/"+'delaytransform'+str(counter2[i])+".png")
		i +=1
	
	#convert output to gif form
	
	images = glob.glob('/data4/paper/rkb/gifstorage/*.png')
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/gifstorage/delaygif.gif', gif,fps=1)
def delaytransformv1(data_dir, stokes):
	if os.path.isdir("/data4/paper/rkb/delaygifstorage/"):
		pass
	else:
		os.makedirs("/data4/paper/rkb/delaygifstorage/")
	#type-abaselines = ['72_112', '97_112', '22_105', '9_88', '9_20', '20_89', '43_89', '53_64', '31_53', '31_65', '80_104', '96_104']
	#type-cbaselines = ['72_105', '88_105', '22_112', '9_22', '9_64', '20_53', '53_80', '10_89', '31_89', '31_104', '43_65', '65_96']
	#baselines = ['64_88', '64_80', '9_105', '9_53', '53_104', '22_72', '20_22', '20_31', '31_96', '65_89', '10_97', '10_43', '72_105', '88_105', '22_112', '9_22', '9_64', '20_53', '53_80', '10_89', '31_89', '31_104', '43_65', '65_96', '72_112', '97_112', '22_105', '9_88', '9_20', '20_89', '43_89', '53_64', '31_53', '31_65', '80_104', '96_104']
	#baselines = ['96_112', '96_105', '64_97', '43_64', '97_104', '65_88', '65_72', '72_104', '10_80', '10_88', '43_105', '80_112']
	# baselines = ['9_105', '22_72', '81_112', '65_89'] #Up1Left1; oneout
	#baselines = ['53_104', '31_96'] #Up1Left1; oneout (must take comp.conj)
	#baselines = ['20_22'] #Up1Left1 ; allin
	#baselines = ['9_53', '20_31', '81_89'] #Up1Left1 ; allin (must take comp.conj)
	baselines = ['22_81', '9_20', '20_89']
	for antstr in baselines:
		ant_i, ant_j = map(int, antstr.split('_'))
		pair = (ant_i, ant_j)
		data, channels = avgfreqcalc(data_dir, antstr, stokes)
		window = aipy.dsp.gen_window(channels, window="blackman-harris")
		d_transform = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(data * window)))
		delays = np.fft.fftshift(np.fft.fftfreq(channels, .1/channels)) # fftfreq takes in (nchan, chan_spacing)
		d_start = delays[0]
		d_end = delays[-1]
		#d_transform = np.abs(d_transform)
		f, ax = plt.subplots(figsize=(5, 4))
		# ax.plot(delays, np.real(np.log(d_transform)), label="real part")
		# ax.plot(delays, np.imag(np.log(d_transform)), label="imag part")
		ax.plot(delays, np.log(np.abs(d_transform)), label="amplitude")
		tauh = calculate_baseline(pair)/2.9979e8*1e9 # convert to ns
		ax = plt.gca()
		ax.axvline(x=0., linestyle='--', color='0.5')
		ax.axvline(x=-tauh, linestyle='--', color='0.5')
		ax.axvline(x=tauh, linestyle='--', color='0.5')
		ax.set_xlim(-400, 400)
		ax.set_xlabel('Delay [bins]')
		ax.set_ylabel('log10(abs(V_I)')
		ax.set_title('Delay Transform'+antstr+stokes)
		plt.legend()
		plt.savefig("/data4/paper/rkb/delaygifstorage/"+'delaytransform'+'{} {}.png'.format(antstr, stokes))
		plt.clf()
	images = glob.glob(('/data4/paper/rkb/delaygifstorage/*.png').format(stokes))
	gif = []
	for filename in images:
		gif.append(imageio.imread(filename))
	imageio.mimsave('/data4/paper/rkb/delayv1lined.gif', gif,fps=1)
	# shutil.rmtree('/data4/paper/rkb/delaygifstorage/')



def delaytransformavgbaseline(data_dir, stokes):
	baselines = ['64_88', '64_80', '9_105', '9_53', '53_104', '22_72', '20_22', '20_31', '31_96', '65_89', '10_97', '10_43', '72_105', '88_105', '22_112', '9_22', '9_64', '20_53', '53_80', '10_89', '31_89', '31_104', '43_65', '65_96', '72_112', '97_112', '22_105', '9_88', '9_20', '20_89', '43_89', '53_64', '31_53', '31_65', '80_104', '96_104']
	avg = 0
	for antstr in baselines:
		ant_i, ant_j = map(int, antstr.split('_'))
		data, channels = avgfreqcalc(data_dir, antstr, stokes)
		window = aipy.dsp.gen_window(channels, window="blackman-harris")
		d_transform = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(data * window)))
		delays = np.fft.fftshift(np.fft.fftfreq(channels, .1/channels))
		avg += d_transform
	avg = avg/len(baselines)
	f, ax = plt.subplots(figsize=(5, 4))
	ax.plot(delays, np.real(np.log10(avg)))
	ax.plot(delays, np.imag(np.log10(avg)))
	ax.set_xlim(-400,400)
	ax.set_ylim(-5, 5)
	ax.set_xlabel('Delay [bins]')
	ax.set_ylabel('log10(V_I)')
	ax.set_title('Delay Transform Averaged over Baseline')
	plt.savefig("/data4/paper/rkb/"+ "delaytransformavged.png")
#Errorlog:
#Error 1: 7/5/17 at 23:51; running into error "UnboundLocalError: local variable 'uv' referenced before assignment"
#Resolved (Error 1): 7/6/17; fixed location of directory; the program wasn't finding anything at the files I pointed it to
#Error 2: 7/7/17 at 9:00; This application failed to start because
#it could not find or load the Qt platform plugin "xcb" in "". Available platform plugins are: minimal, offscreen, xcb. Reinstalling the application may fix this problem. Occured when running in folio.



