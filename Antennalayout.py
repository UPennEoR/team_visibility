from pyuvdata import UVdata
UV = UVdata()

def baselinecategorizer(baselinetype, outin):
	#14.6m
	if baselinetype = 'Up1Right1':
		if outin = "Allin":
			baselinelist = ['9_22', '20_81', '31_89', '53_20']
		elif outin = "Oneout":
			baselinelist = ['64_9', '80_53', '104_31', '89_10', '81_97', '22_112']
		else:
			baselinelist = ['105_72', '88_105', '96_65', '65_43']
	elif baselinetype = 'Up1Left1':
		if outin="Allin":
			baselinelist = ['53_9', '31_20', '89_81', '20_22']
		elif outin = "Oneout":
			baselinelist = ['9_105', '22_72', '81_112', '65_89', '96_31', '104_53']
		else:
			baselinelist = ['64_88', '80_64', '43_10', '10_97']
	elif baselintype="Right1":
		if outin="Allin":
			baselinelist=['22_81', '9_20', '20_89', '53_31']
		elif outin = "Oneout":
			baselinelist=['105_22', '81_10', '88_9', '89_43', '64_53', '31_65']
		else:
			baselinelist=['72_112', '112_97', '80_104','104_96']
	elif baselinetype="Left1":
		if outin="Allin":
			baselinelist=['81_22', '20_9', '89_20', '31_53']
		elif outin="Oneout":
			baselinelist=['22_105', '10_81', '9_88', '43_89', '53_64', '65_31']
		else:
			baselinelist=['97_112', '112_72', '96_104', '104_80']
	#29.2m
	elif baselinetype="Up2Left2":
		if outin = "Allin":
			baselinelist=['31_22']
		elif outin="Oneout":
			baselinelist=['53_105', '20_72', '89_112', '65_81', '96_20', '104_9']
		else:
			baselinelist = ['80_88', '43_97']
	elif baselinetype="Up2Right2":
		if outin="Allin":
			baselinelist=['51_81']
		elif outin="Oneout":
			baselinelist=['9_112', '20_97', '31_10', '104_89', '80_20', '64_22']
		else:
			baselinelist=['88_72', '96_43']
	elif baselinetype = 'Right2':
		if outin = "Allin":
			baselinelist=['9_89']
		elif outin = "Oneout":
			baselinelist=['105_81', '22_10', '88_20', '20_43', '64_31', '53_65']
		else:
			baselinelist=['72_97', '80_96']
	elif baselinetype='Left2':
		if outin="Allin":
			baselinelist= ['89_9']
		elif outin = "Oneout":
			baselinelist= ['81_105','10_22','43_20', '20_88', '65_53', '31_64']
		else:
			baselinelist=['97_72', '96_80']
	#43.8 m
	elif baselinetype = 'Right3':
		if outin="Allin":
			baselinelist=[]
		elif outin= "Oneout":
			baselinelist = ['88_89', '9_43']
		else:
			baselinelist=['105_10', '64_65']
	elif baselinetype= "Left3":
		if outin="Allin":
			baselinelist = []
		elif outin="Oneout":
			baselinelist=['43_9', '89_88']
		else:
			baselinelist = ['10_105', '65_64']


	data = UV.get_data(baselinelist)
	return data