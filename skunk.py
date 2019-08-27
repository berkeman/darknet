
class nndata():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')

	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.h_in, self.w_in, self.c_in, self.groups, self.k, self.c_out, self.stride = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		self.h_ut, self.w_ut, self.c_ut, self.pad = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')

		def getdata():
			return tuple(map(float, self.fh.readline().rstrip().split(' ')))

		self.weights = getdata()
		self.biases  = getdata()
		self.inputs  = getdata()
		self.outputs = getdata()



def checkconvlayer(N):
	print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print(N.k, N.groups, N.stride, N.pad)

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut
	G = N.groups
	P = N.pad
	K = N.k

	# @@@ Does not handle groups

	errs = 0
	maxerr = 0

	for h in range(0, N.h_in, N.stride):
		for w in range(0, N.w_in, N.stride):
			for cu in range(0, CU):
				for g in range(0, N.groups):
					t = 0.0
					for ci in range(0, CI//G):
						for y in range(-(K//2), K//2 + 1):
							for x in range(-(K//2), K//2 + 1):
								ww = w + x
								hh = h + y
								if ww < 0 or ww >= W: continue
								if hh < 0 or hh >= H: continue

								six =  ww     + W*hh      + W*H*ci + W*H*CI//G*g
								wix = (x + P) + K*(y + P) + K*K*ci + K*K*CI//G*cu
								tix = (w//N.stride) + (h//N.stride)*N.w_ut + (H*W//N.stride//N.stride)*cu
								t += N.inputs[six] * N.weights[wix]
#								print('b %6d %6d %6d' % (six, wix, tix,))
				yy = N.outputs[tix]
				if abs(t - yy) >= 1e-5:
					print(w, h, '  ', x, y, '  ', ci, cu, '     ', t, yy)
					errs += 1
					maxerr = max(maxerr, abs(t-yy))
	print('errs', errs)
	print('maxerr', maxerr)

N = nndata('darknet_run.txt')

for x in range(60):
	print(x)
	N.nextlayer()
	checkconvlayer(N)
