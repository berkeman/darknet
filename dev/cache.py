class CacheMissException(Exception):
	pass

class StupidCache():
	def __init__(self, size):
		print('StupidCache', size)
		self.m = {}
		self.size = size
		self.ts = 0
		self.hits = 0
		self.miss = 0
		self.reads = 0

	def write(self, a, d):
#		assert isinstance(a, int)
		if len(self.m) == self.size:
			# is full
			if a in self.m:
				# overwrite existing
				self.m[a] = (d, self.ts)
			else:
				# overwrite oldest item
				oldest = min(self.m.keys(), key=lambda x: self.m[x][1])
				self.m.pop(oldest)
#				print('o', oldest)
				self.m[a] = (d, self.ts)
		else:
			# insert or replace
			self.m[a] = (d, self.ts)
		self.ts += 1

	def read(self, a):
		self.reads += 1
		if a not in self.m:
			self.miss += 1
			raise CacheMissException
		else:
			self.hits += 1
			return self.m[a][0]
