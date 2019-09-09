
datasets=('source',)

def synthesis():
	max_twolinecache = 0
	max_inputfeature = 0
	max_dwcache9     = 0

	for typ, sx, sy, stride, wi, hi, ci, wo, ho, co in datasets.source.iterate(None, ('type', 'sx', 'sy', 'stride', 'wi', 'hi', 'ci', 'wo', 'ho', 'co',)):
		if typ=='conv' and sx==sy==1 and stride==1:
			twolinecache = wi*2*ci
			inputfeature = wi*hi*ci
			dwcache9     = 3*3*co
			max_twolinecache = max(max_twolinecache, twolinecache)
			max_inputfeature = max(max_inputfeature, inputfeature)
			max_dwcache9     = max(max_dwcache9, dwcache9)
			if co > ci:
				print('Expansion layer')
				print("  %dx%d-%d -> %dx%d-%d" % (wi, hi, ci, wo, ho, co))
				print('  input feature size (words)    %9d' % (inputfeature,))
				print('  two-line input cache (words)  %9d' % (twolinecache,))
				print('  3x3 output cache (words)      %9d' % (dwcache9,))
				print()

	print('max_inputfeature %9d' % (max_inputfeature,))
	print('max_twolinecache %9d' % (max_twolinecache,))
	print('max_dwcache9,    %9d' % (max_dwcache9))
