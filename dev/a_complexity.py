from collections import namedtuple

datasets=('source',)

Layer = namedtuple('Layer', 'layer sx sy groups stride wi hi ci wo ho co')
ResLayer = namedtuple('ResLayer', 'wi hi ci cm co stride')

def window(v, N=3):
	# return tuple (x_n, x_{n+1}, x_{n+N-1}) for each n.
	# NB: Does not return head or tail.
	buf = []
	for n, item in enumerate(v):
		buf.append(item)
		buf = buf[-N:]
		if len(buf)==N:
			yield (n, buf)


def synthesis():
	max_twolinecache = 0
	max_inputfeature = 0
	max_dwcache9     = 0

	v = []
	for data in datasets.source.iterate(None, ('layer', 'sx', 'sy', 'groups', 'stride', 'wi', 'hi', 'ci', 'wo', 'ho', 'co',)):
		v.append(Layer(*data))


	resv = []
	pops = set()
	for n, (l0, l1, l2) in window(v, 3):
		if (l0.ci <= l1.ci >= l2.co) and (l0.sx==l0.sy==l2.sx==l2.sy==1) and (l1.sx==l1.sy==3) and (l1.groups==l1.ci==l1.co):
			#print("Bottleneck:   %3d -> %3d -> %3d  (stride %s)" % (l0.ci, l1.ci, l2.co, l1.stride))
			resv.append(ResLayer(l0.wi, l0.hi, l0.ci, l1.ci, l2.co, l1.stride))
			pops.update({n-2, n-1, n})

	print('Bottlenecks:')
	for item in resv:
		print(item)


	print('Remaining layers:')
	v = [x for n, x in enumerate(v) if n not in pops]
	for item in v:
		print(item)



	for l in v:
		if l.layer=='conv' and l.sx==l.sy==1 and l.stride==1:
			twolinecache = l.wi*2*l.ci
			inputfeature = l.wi*l.hi*l.ci
			dwcache9     = 3*3*l.co
			max_twolinecache = max(max_twolinecache, twolinecache)
			max_inputfeature = max(max_inputfeature, inputfeature)
			max_dwcache9     = max(max_dwcache9, dwcache9)
			if l.co > l.ci:
				print('Expansion layer')
				print("  %dx%d-%d -> %dx%d-%d (%dx%d)" % (l.wi, l.hi, l.ci, l.wo, l.ho, l.co, l.sx, l.sy))
				print('  input feature size (words)    %9d' % (inputfeature,))
				print('  two-line input cache (words)  %9d' % (twolinecache,))
				print('  3x3 output cache (words)      %9d' % (dwcache9,))
		elif l.layer=='conv':
			print('Other conv')
			print("  %dx%d-%d -> %dx%d-%d (%dx%d)" % (l.wi, l.hi, l.ci, l.wo, l.ho, l.co, l.sx, l.sy))
		print()


	print('max_inputfeature %9d' % (max_inputfeature,))
	print('max_twolinecache %9d' % (max_twolinecache,))
	print('max_dwcache9,    %9d' % (max_dwcache9))
