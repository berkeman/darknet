
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
	S = N.stride

	# @@@ Does not handle groups

	errs = 0
	maxerr = 0

	if G == 1:
		for h in range(0, N.h_in, S):
			for w in range(0, N.w_in, S):
				for cu in range(0, CU):
					t = 0.0
					for ci in range(0, CI//G):
						for y in range(-(K//2), K//2 + 1):
							for x in range(-(K//2), K//2 + 1):
								ww = w + x
								hh = h + y
								if ww < 0 or ww >= W: continue
								if hh < 0 or hh >= H: continue

								six =  ww     + W*hh      + W*H*ci
								wix = (x + P) + K*(y + P) + K*K*ci + K*K*CI*cu
								tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
								t += N.inputs[six] * N.weights[wix]
#								print('b %6d %6d %6d     %2d %2d %2d' % (six, wix, tix, g, ci, cu))
					yy = N.outputs[tix]
					if abs(t - yy) >= 1e-5:
						print(w, h, '  ', x, y, '  ', ci, cu, '     ', t, yy)
						errs += 1
						maxerr = max(maxerr, abs(t-yy))
	elif G == CI == CU:
		pass
	else:
		assert False


def convlayer(N):
	print(N.w_in, N.h_in, N.c_in, '->', N.w_ut, N.h_ut, N.c_ut)
	print(N.k, N.groups, N.stride, N.pad)

	W = N.w_in
	H = N.h_in
	CI = N.c_in
	CU = N.c_ut
	G = N.groups
	P = N.pad
	K = N.k
	S = N.stride

	out = [0. for _ in range(H*W*CU//S//S)]

	if G == 1:
		for h in range(0, H, S):
			for w in range(0, W, S):
				for cu in range(0, CU):
					t = 0.0
					for ci in range(0, CI):
						for y in range(-(K//2), K//2 + 1):
							for x in range(-(K//2), K//2 + 1):
								ww = w + x
								hh = h + y
								if ww < 0 or ww >= W: continue
								if hh < 0 or hh >= H: continue

								six =  ww     + W*hh      + W*H*ci
								wix = (x + P) + K*(y + P) + K*K*ci + K*K*CI*cu
								tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
								t += N.inputs[six] * N.weights[wix]
					out[tix] = t
	elif G == CI == CU:
		for h in range(0, H, S):
			for w in range(0, W, S):
				ci = 0
				for cu in range(0, CU):
					t = 0.0
					for y in range(-(K//2), K//2 + 1):
						for x in range(-(K//2), K//2 + 1):
							ww = w + x
							hh = h + y
							if ww < 0 or ww >= W: continue
							if hh < 0 or hh >= H: continue

							six =  ww     + W*hh      + W*H*cu
							wix = (x + P) + K*(y + P) + K*K*ci + K*K*cu
							tix = (w//S) + (h//S)*N.w_ut + (H*W//S//S)*cu
#							print('dw %6d %6d %3d %3d   %3d %3d   %3d %3d   %f %f' % (six, wix, w, h, ww, hh, ci, cu, N.inputs[six], N.weights[wix]))
							t += N.inputs[six] * N.weights[wix]
#					print(t, N.outputs[tix])
					out[tix] = t
		pass
	else:
		assert False
	return out



def check(out, N):
	errs = 0
	maxerr = 0
	print(N.h_ut*N.w_ut*N.c_ut)
	for ix in range(N.h_ut*N.w_ut*N.c_ut):
		golden = N.outputs[ix]
		pred = out[ix]
		diff = abs(golden - pred)
		if diff >= 1e-5:
#			print(ix, golden, pred, diff)
			errs += 1
			maxerr = max(maxerr, diff)

	print('errs', errs)
	print('maxerr', maxerr)

N = nndata('darknet_run.txt')

for x in range(60):
	print(x)
	N.nextlayer()
	print('conv')
	out = convlayer(N)
	print('check')
	check(out, N)
	print('')
