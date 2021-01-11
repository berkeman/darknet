from os.path import join
from os import symlink
import subprocess
from glob import glob

depend_extra = ('../darknet',)


def synthesis(job):
	symlink(join(job.input_filename("mobilenet")), "mobilenet")
	symlink(join(job.input_filename("data")), "data")
	command = \
	   job.input_filename("darknet") + " classifier predict " + \
	   job.input_filename("cfg/imagenet1k.data ") + \
	   "mobilenet/test.cfg mobilenet/test.weights mobilenet/cat.jpg"

	with job.open('command.txt', 'wt') as fh:
		fh.write(command)

	subprocess.check_call(command.split())

	for item in glob('data_layer*.txt'):
		job.register_file(item)
