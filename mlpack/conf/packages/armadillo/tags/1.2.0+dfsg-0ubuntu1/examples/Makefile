# The generated "Makefile" from "Makefile.cmake" is only usable after
# the Armadillo library has been configured and installed by CMake.

CXX=g++
#CXX=g++-4.2
## Under MacOS you may have an old compiler as default (e.g. GCC 4.0).
## However, GCC 4.2 or later is available and preferable due to better
## handling of template code.

#CXX=CC
## When using the Sun Studio compiler


# flags configured by CMake
ifeq (unix,macos)
  EXTRA_LIB_FLAGS = -framework Accelerate
endif

#EXTRA_LIB_FLAGS = -library=sunperf
## When using the Sun Studio compiler


ifeq (true,true)
  BOOST_INCLUDE_FLAG = -I /usr/include
endif



LIB_FLAGS = -larmadillo $(EXTRA_LIB_FLAGS)
## NOTE: on Ubuntu and Debian based systems you may need to add 
## -lgfortran to LIB_FLAGS



OPT = -O1
## As the Armadillo library uses recursive templates,
## compilation times depend on the level of optimisation:
##
## -O0: quick compilation, but the resulting program will be slow
## -O1: produces programs which achieve most of the possible speedup
## -O3: produces programs which have almost all possible speedups,
##      but compilation takes considerably longer


#OPT = -xO4 -xannotate=no
## When using the Sun Studio compiler


#EXTRA_OPT = -fwhole-program
## Uncomment the above line if you're compiling 
## all source files into one program in a single hit.


#DEBUG = -DARMA_EXTRA_DEBUG
## Uncomment the above line to enable low-level
## debugging.  Lots of debugging information will
## be printed when a compiled program is run.
## Please enable this option when reporting bugs.


#FINAL = -DARMA_NO_DEBUG
## Uncomment the above line to disable Armadillo's checks.
## DANGEROUS!  Not recommended unless your code has been
## thoroughly tested.


#
#
#

CXXFLAGS = $(BOOST_INCLUDE_FLAG) $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: example1 example2

example1: example1.cpp
	$(CXX) $(CXXFLAGS)  -o $@  $<  $(LIB_FLAGS)

example2: example2.cpp
	$(CXX) $(CXXFLAGS)  -o $@  $<  $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f example1 example2

