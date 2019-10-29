from accelerator.dataset import DatasetWriter

datasets=('reslayers',)

def synthesis():

	dw = DatasetWriter(parent=datasets.reslayers)
	dw.add('name', 'unicode')
	dw.add('inputfeatsize', 'number')
	dw.add('outputfeatsize', 'number')
	dw.add('twolinecache', 'number')
	dw.add('dwcache9', 'number')
	dw.set_slice(0)

	for wi, hi, ci, cm, co, wo, ho in datasets.reslayers.iterate(None, ('wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho',)):
		name = "%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (wi, hi, ci,   wi, hi, cm,   wo, ho, co,)
		inputfeature  = wi * hi * ci
		outputfeature = wo * ho * co
		twolinechache = wi * 2 * ci
		dwcache9      = 9 * cm
		dw. write(name, inputfeature, outputfeature, twolinechache, dwcache9)
