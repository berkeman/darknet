from collections import namedtuple

datasets=('source',)

Layer = namedtuple('Layer', 'layer sx sy groups stride wi hi ci wo ho co')
ResLayer = namedtuple('ResLayer', 'wi hi wo ho ci cm co stride')

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

	v = []
	for data in datasets.source.iterate(None, ('layer', 'sx', 'sy', 'groups', 'stride', 'wi', 'hi', 'ci', 'wo', 'ho', 'co',)):
		v.append(Layer(*data))

	reslayers = []
	pops = set()
	for n, (l0, l1, l2) in window(v, 3):
		if (l0.ci <= l1.ci >= l2.co) and (l0.sx==l0.sy==l2.sx==l2.sy==1) and (l1.sx==l1.sy==3) and (l1.groups==l1.ci==l1.co):
			#print("Bottleneck:   %3d -> %3d -> %3d  (stride %s)" % (l0.ci, l1.ci, l2.co, l1.stride))
			reslayers.append(ResLayer(l0.wi, l0.hi, l2.wo, l2.ho, l0.ci, l1.ci, l2.co, l1.stride))
			pops.update({n-2, n-1, n})

	with open('result.txt', 'wt') as fh:
		fh.write('Bottlenecks:\n')
		for item in reslayers:
			fh.write('  ' + str(item) + '\n')
		fh.write('Remaining layers:\n')
		remlayers = [x for n, x in enumerate(v) if n not in pops]
		for item in remlayers:
			fh.write('  ' + str(item) + '\n')

	return reslayers, remlayers
