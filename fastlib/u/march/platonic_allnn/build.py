# a librule creates a library, there must not be a main
# It is possible to have many librules in a single build.py
librule(
	# What do you want the library to be called?
	# Use this name to include the library in binrules elsewhere
	# If the name is omitted, the library will be called the name of the directory
	name = "allnn",
	
	# Any .c or .cc files where library functions are defined
	# This line can be omitted if there are no .cc or .c files
	#sources = ["allnn.cc"],

	# Any .h files where library functions are defined
	headers = ["allnn.h"],

	# libraries this library depends on
	# fastlib:fastlib means the fastlib library is in the fastlib directory
	# fastlib includes all the library functionality		
	deplibs = ["fastlib:fastlib"],

	# A file containing a main with test functions
	# fl-build allnn_tests creates an executable called allnn_tests
	#tests = ["allnn_tests.cc"]
)



# a binrule creates an executable, there must be a main in one of the sources
# It is possible to have many binrules in a single build.py file
binrule(
	
	# the name of the executable
	name = "allnn_main",

	sources = ["allnn_main.cc"],

	# This line can be omitted if there are no headers
	#headers = ["allnn_main.h"],

	# :allnn means allnn is in the same directory as this build.py file
	deplibs = [":allnn", "fastlib:fastlib"]
)
