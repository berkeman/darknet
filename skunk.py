
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
	C = N.c_in
	G = N.groups
	P = N.pad
	K = N.k

	# @@@ Does not handle groups

	errs = 0
	maxerr = 0

	for h in range(0, N.h_in, N.stride):
		for w in range(0, N.w_in, N.stride):
			for g in range(0, N.groups):
				t = 0.0
				for c in range(0, C//G):
					for y in range(-(K//2), K//2 + 1):
						for x in range(-(K//2), K//2 + 1):
							ww = w + x
							hh = h + y
							if ww < 0 or ww >= W: continue
							if hh < 0 or hh >= H: continue

							six =  ww     + W*hh      + W*H*(c + g*C//G)
							wix = (x + P) + K*(y + P) + K*K*(c + g*C//G)
							tix = w//N.stride + (h//N.stride)*N.w_ut + g*H*W//N.stride//N.stride
							t += N.inputs[six] * N.weights[wix]
				yy = N.outputs[tix]
#				print(w, h, '-', c, g, '-', t, yy, t-yy, abs(t-yy) < 1e-5)
				if abs(t - yy) >= 1e-5:
					errs += 1
				maxerr = max(maxerr, abs(t-yy))
	print('errs', errs)
	print('maxerr', maxerr)

N = nndata('darknet_run.txt')

for x in range(60):
	print(x)
	N.nextlayer()
	checkconvlayer(N)
