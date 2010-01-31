# A librule creates a library, or code that lacks a main function.
# You can define many librules in a single build.py.
librule(
	# What do you want the library to be called?  You'll use this
	# name to include the library in binrules elsewhere.  If the
	# omitted, the librule uses the name of build.py's directory.
	name = "allnn",
	
	# Any .c or .cc files where library functions are defined.
	# This line can be omitted if there are no .cc or .c files.
	#sources = ["allnn.cc"],
	
	# Any .h files where library functions are defined.
	headers = ["allnn.h"],
	
	# Other libraries that this library depends upon.  It's often
	# easiest just to indicate "fastlib:fastlib", interpreted
	# "directory:librule", to link with all of FASTlib's core
	# components.
	deplibs = ["fastlib:fastlib"],
	
	# You can specify a unit test file, which should contain a
	# main function that runs a batch of tests.  In the future,
	# this will be compiled and run automatically, but for now,
	# you can compile this explicitly with "fl-build allnn_tests".
	#tests = ["allnn_tests.cc"]
)



# A binrule creates an executable, or a stand-alone program that has a
# main function.  It's possible to have many binrules in one build.py.
binrule(
	
	# The name of the executable.
	name = "allnn_main",
	
	# The .c or .cc file containing main and any others you need.
	sources = ["allnn_main.cc"],
	
	# This line can be omitted if there are no headers.
	#headers = ["allnn_main.h"],
	
	# The leading colon means to check this build.py for allnn.
	deplibs = [":allnn"]
)
