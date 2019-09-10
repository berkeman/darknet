from math import ceil

class Memory():
	def __init__(self, naddr, nwords, verbose=False):
		print('[Memory create: %d x %d]' % (naddr, nwords,))
		self.m = [[0 for x in range(nwords)] for y in range(naddr)]
		self.naddr = naddr
		self.nwords = nwords
		self.verbose = verbose
		if self.verbose:
			self.printsize()

	def write(self, a, d):
		assert len(d) == self.nwords
		assert 0 <= a <= self.naddr
		self.m[a] = d
#		print('write %5d' % (a,), ''.join("% 5.2f" % x for x in d))

	def read(self, a):
		assert 0 <= a <= self.naddr
		return self.m[a]

	def printsize(self):
		print('Memory naddr: ', self.naddr)
		print('Memory nwords:', self.nwords)


def feat2mem(mem, data, width, height, channels):
	"""
	Store "data" in "mem" as a cube "width"x"height"x"channels".
	There are "mem.nwords" channels per address.
	"""
	assert width * height * channels // mem.nwords <= mem.naddr
	depth = ceil(channels / mem.nwords)
	for y in range(height):
		for x in range(width):
			for chigh in range(depth):
				dstadr = x + y * width + chigh * width * height
				word = [0. for _ in range(mem.nwords)]
				for clow in range(mem.nwords):
					if chigh * mem.nwords + clow < channels:
						srcadr = x + y * width + (chigh * mem.nwords + clow) * width * height
						word[clow] = data[srcadr]
					else:
						word[clow] = 0.0
				mem.write(dstadr, word)
	# check
	for y in range(height):
		for x in range(width):
			for c in range(channels):
				goladr = x + y * width + c * width * height
				futadr = x + y * width + (c // mem.nwords) * width * height
				golden = data[goladr]
				fut = mem.read(futadr)[c % mem.nwords]
				assert fut == golden


def test_feat2mem():
	# import importlib
	# importlib.reload(memory)
	# memory.test_feat2mem()
	mem = Memory(2*3*5, 2, verbose=True)
	data = list(range(2*3*5))
	feat2mem(mem, data, 2, 3, 5)
	for a in range(mem.naddr):
		print(a, mem.m[a])


def create_weight_mem_1x1(weights, nwords, channels_in, channels_out):
	WL = nwords
	CI = channels_in
	CU = channels_out
	# create and fill weigth memory
	inblocks = ceil(CI/WL)
	mem = Memory(inblocks * CU, WL)
	for cu in range(CU):
		for ci in range(inblocks):
			data = []
			for c in range(WL):
				srcadr = ci * WL + cu * CI + c
				if srcadr < channels_in * channels_out:
					data.append(weights[srcadr])
				else:
					data.append(0.)
			mem.write(ci + cu*inblocks, data)
	# check
	for ci in range(CI):
		for cu in range(CU):
			golden = weights[ci + cu*CI]
			wut = mem.read(ci // WL + cu * inblocks)[ci % WL]
			assert golden == wut
	return mem


#if  __name__ == 'main':
	
