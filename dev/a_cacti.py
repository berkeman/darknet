from os.path import join
import subprocess

options = {
	"args": {},
}

depend_extra = ("../cacti.5.3.rev.174/cacti",)

ARG_ORDER = "C B A RWP ERP EWP NSER NBANKS TECH OUTPUT_WIDTH CUSTOM_TAG TAG_WIDTH ACCESS_MODE PLAIN_RAM DRAM OPT_DYN_ENERGY OPT_DYN_POWER OPT_LEAK_POWER OPT_RAND_CYCLE_TIME TEMPERATURE DATA_ARRAY_CELL_DEVICE_TYPE DATA_ARRAY_PERIPH_DEVICE_TYPE TAG_ARRAY_CELL_DEVICE_TYPE TAG_ARRAY_PERIPH_DEVICE_TYPE INTERCONNECT_PROJECTION_TYPE WIRE_TYPE_INSIDE_MAT WIRE_TYPE_OUTSIDE_MAT REPEATERS_IN_HTREE VERT_HTREE_WIRES_OVER_THE_ARRAY BROADCAST_ADDR_DATA_OVER_VERT_HTREE MAX_AREA_CONSTRAINT MAX_ACC_TIME_CONSTRAINT MAX_REPEATER_DELAY_CONSTRAINT PAGE_SIZE BURST_LENGTH INTERNAL_PREFETCH".split()

DEFAULT_ARGS = {
	"C": 1024,
	"B": 1,
	"A": 1,
	"RWP": 0,
	"ERP": 0,
	"EWP": 1,
	"NSER": 1,
	"NBANKS": 1,
	"TECH": 32,
	"OUTPUT_WIDTH": 8,
	"CUSTOM_TAG": 0,
	"TAG_WIDTH": 0,
	"ACCESS_MODE": 0,
	"PLAIN_RAM": 0,
	"DRAM": 0,
	"OPT_DYN_ENERGY": 0,
	"OPT_DYN_POWER": 0,
	"OPT_LEAK_POWER": 0,
	"OPT_RAND_CYCLE_TIME": 0,
	"TEMPERATURE": 360,
	"DATA_ARRAY_CELL_DEVICE_TYPE": 0,
	"DATA_ARRAY_PERIPH_DEVICE_TYPE": 0,
	"TAG_ARRAY_CELL_DEVICE_TYPE": 0,
	"TAG_ARRAY_PERIPH_DEVICE_TYPE": 0,
	"INTERCONNECT_PROJECTION_TYPE": 1,
	"WIRE_TYPE_INSIDE_MAT": 1,
	"WIRE_TYPE_OUTSIDE_MAT": 1,
	"REPEATERS_IN_HTREE": 1,
	"VERT_HTREE_WIRES_OVER_THE_ARRAY": 0,
	"BROADCAST_ADDR_DATA_OVER_VERT_HTREE": 0,
	"MAX_AREA_CONSTRAINT": 50,
	"MAX_ACC_TIME_CONSTRAINT": 10,
	"MAX_REPEATER_DELAY_CONSTRAINT": 10,
	"PAGE_SIZE": 0,
	"BURST_LENGTH": 1,
	"INTERNAL_PREFETCH": 1,
}

assert len(set(ARG_ORDER)) == len(ARG_ORDER)
assert set(ARG_ORDER) == set(DEFAULT_ARGS)

def synthesis(SOURCE_DIRECTORY):

	command = [join(SOURCE_DIRECTORY, "./cacti.5.3.rev.174/cacti")]
	for arg in ARG_ORDER:
		command.append(str(options.args.get(arg, DEFAULT_ARGS[arg])))
	unknown_args = set(options.args) - set(DEFAULT_ARGS)
	assert not unknown_args, "Unknown args: " + ", ".join(sorted(unknown_args))

	subprocess.check_call(command)

	with open('command.txt', 'wt') as fh:
		fh.write(' '.join(command) + '\n')

	with open("out.csv") as fh:
		keys = fh.readline().strip().split(", ")
		values = fh.readline().strip().split(", ")
	def parser():
		for v in values:
			try:
				yield int(v)
			except ValueError:
				yield float(v)
			except ValueError:
				yield v
	return dict(zip(keys, parser()))
