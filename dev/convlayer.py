from math import ceil

def conv1x1_block(xmem, ymem, wmem, width, height, channels_in, channels_out):
	assert xmem.nwords == ymem.nwords == wmem.nwords
	assert width * height * channels_in  // xmem.nwords <= xmem.naddr
	assert width * height * channels_out // ymem.nwords <= ymem.naddr

	print('conv1x1_block')

	WL = xmem.nwords

	inblocks = ceil(channels_in/WL)
	utblocks = ceil(channels_out/WL)

	print('inblocks', inblocks)
	print('utblocks', utblocks)

	def blockdotprod(xv, fv):
		s = 0
		for x, f in zip(xv, fv):
			s += sum(xi * fi for xi, fi in zip(x, f))
		return s

	for h in range(height):
		for w in range(width):
			x = tuple(xmem.read(w + h*width + c*width*height) for c in range(inblocks))  # fetch all input channels
			for chigh in range(utblocks):
				t = []
				for clow in range(WL):
					cu = chigh*WL + clow # c is output channel
					if cu < channels_out:
						f = tuple(wmem.read(cu*inblocks + c) for c in range(inblocks))
						t.append(blockdotprod(x, f))
					else:
						t.append(0)
				ymem.write(w + h*width + chigh*width*height, t)


def conv3x3dw_block(xmem, ymem, wmem, width, height, channels, OUTPUTS):
	assert xmem.nwords == ymem.nwords == wmem.nwords
	assert width * height * channels // xmem.nwords <= xmem.naddr
	assert width * height * channels // ymem.nwords <= ymem.naddr

	print('conv3x3dw_block')

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
#						print("%3d %3d %3d  % 1d % 1d %6d" % (w, h, chigh, x, y, wadr))
						if x + w < 0 or x + w >= width or y + h < 0 or y + h >= height:
#							print('pad data')
							data = [0 for _ in range(WL)]
						else:
							data = xmem.read(srcadr)
						weight = wmem.read(wadr)
#						print("%3d %3d   %3d %3d   %5.2f %5.2f  (%5d %5d)" % (w, h, x, y, data[1], weight[1], srcadr, wadr))
						acc = [ta + tw * tx for ta, tw, tx in zip(acc, weight, data)]
#				print('x', w, h, OUTPUTS[w+h*width+width*height], acc[1])
				ymem.write(w + h * width + chigh * width * height, acc)
#		if h > 1:
#			exit()
