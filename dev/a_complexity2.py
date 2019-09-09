datasets=('reslayers',)

def synthesis():

	print(datasets)

	for wi, hi, ci, cm, co, wo, ho in datasets.reslayers.iterate(None, ('wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho',)):
		print()
		print("%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (
			wi, hi, ci,
			wi, hi, cm,
			wo, ho, co,
		))
		inputfeature  = wi * hi * ci
		twolinechache = wi * 2 * ci
		dwcache9      = 9 * cm
		print(inputfeature, twolinechache, dwcache9)
