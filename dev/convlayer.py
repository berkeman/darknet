from math import ceil, sqrt

class Blockdotprod():
	def __init__(self):
		self.mulcnt = 0
		self.cnt = 0
		pass
	def mac(self, xvec, kvec):
		s = 0
		for x, k in zip(xvec, kvec):
			s += sum(xi * ki for xi, ki in zip(x, k))
		self.mulcnt += sum(len(x) for x in xvec)
		self.cnt += 1
		return s
	def status(self):
		return(dict(mulcnt=self.mulcnt, cnt=self.cnt))

class BiasNorm():
	def __init__(self, layer):
		self.mean = layer.rolling_mean
		self.var  = layer.rolling_var
		self.bias = layer.biases
		self.scales = layer.scales
	def bn(self, x, channel):
		# Bias and (batch) normalisation
		# x = input value
		x = (x - self.mean[channel]) / sqrt(self.var[channel] + 1e-9)
		x = x * self.scales[channel]
		x = x + self.bias[channel]
		return x



bdp = Blockdotprod()

def conv1x1_block(xmem, ymem, wmem, width, height, channels_in, channels_out, bias):
	assert xmem.nwords == ymem.nwords == wmem.nwords
	assert width * height * channels_in  // xmem.nwords <= xmem.naddr
	assert width * height * channels_out // ymem.nwords <= ymem.naddr

	print('conv1x1_block %dx%d %d->%d' % (width, height, channels_in, channels_out))

	WL = xmem.nwords

	inblocks = ceil(channels_in/WL)
	utblocks = ceil(channels_out/WL)

	print('inblocks', inblocks)
	print('utblocks', utblocks)

	for h in range(height):
		for w in range(width):
			x = tuple(xmem.read(w + h*width + c*width*height) for c in range(inblocks))  # fetch all input channels
			for chigh in range(utblocks):
				t = []
				for clow in range(WL):
					cu = chigh*WL + clow # c is output channel
					if cu < channels_out:
						f = tuple(wmem.read(cu*inblocks + c) for c in range(inblocks)) # fetch all coeff for one output channel
						y = bdp.mac(x, f)
						y = bias.bn(y, cu)
						t.append(y)
					else:
						t.append(0)
				ymem.write(w + h*width + chigh*width*height, t)



class macbox_1x1():
	def __init__(self, wordlen):
		self.wordlen = wordlen
	def mac(self, x, k, acc):
		# acc += x*k
		assert len(x) == len(k) == self.wordlen


def conv3x3dw_block(xmem, ymem, wmem, width, height, channels, OUTPUTS):
	assert xmem.nwords == ymem.nwords == wmem.nwords
	assert width * height * channels // xmem.nwords <= xmem.naddr
	assert width * height * channels // ymem.nwords <= ymem.naddr

	print('conv3x3dw_block %dx%d %d' % (width, height, channels))

	K = 3
	K2 = K//2

	WL = xmem.nwords

	chanblocks = ceil(channels / WL)

	print('chanblocks', chanblocks)

	for h in range(height):
		for w in range(width):
			for chigh in range(chanblocks):
				acc = [0 for _ in range(WL)]
				for y in range(-K2, K2+1):
					for x in range(-K2, K2+1):
						srcadr = (w + x) + (h + y) * width + chigh * width * height
						wadr = (x+K2) + (y+K2) * K + chigh * K * K
						if x + w < 0 or x + w >= width or y + h < 0 or y + h >= height:
							data = [0 for _ in range(WL)]
						else:
							data = xmem.read(srcadr)
						weight = wmem.read(wadr)
						acc = [ta + tw * tx for ta, tw, tx in zip(acc, weight, data)]
				ymem.write(w + h * width + chigh * width * height, acc)
