from os.path import join
from os import symlink

depend_extra = ('../darknet',)

import subprocess

def synthesis(SOURCE_DIRECTORY):

	symlink(join(SOURCE_DIRECTORY, "mobilenet"), "mobilenet")
	symlink(join(SOURCE_DIRECTORY, "data"), "data")

	command = \
		join(SOURCE_DIRECTORY, "./darknet") + " classifier predict " + \
		join(SOURCE_DIRECTORY, "cfg/imagenet1k.data ") + \
		"mobilenet/test.cfg mobilenet/test.weights mobilenet/cat.jpg"

	print( command )

	subprocess.run(command.split())

