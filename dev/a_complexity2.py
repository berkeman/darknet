datasets=('reslayers',)

def synthesis():

	print(datasets)

	m = 0

	for wi, hi, ci, cm, co, wo, ho in datasets.reslayers.iterate(None, ('wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho',)):
		print()
		print("%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (wi, hi, ci,   wi, hi, cm,   wo, ho, co,))
		inputfeature  = wi * hi * ci
		twolinechache = wi * 2 * ci
		dwcache9      = 9 * cm
		print(inputfeature, twolinechache, dwcache9)

		mac1 = wi*hi * ci * cm
		mac2 = wo*ho *  9 * cm
		mac3 = wo*ho * cm * co
		print(mac1, mac2, mac3, mac1+mac2+mac3)
		m += mac1 + mac2 + mac3

	print(m)
