#
# Uncomment this line to specify the C compiler if the system default isn't
# what you want to use.  
#
#CC=cc
#
# Use this line to specify options for the C compiler.  You'll probably
# want to turn on optimizations. You may also have to use some of the 
# following flags:
#
#  -DCAPSBLAS         if BLAS routine names are capitalized.
#  -DCAPSLAPACK       if LAPACK routine names are capitalized.
#  -DNOUNDERBLAS      if BLAS routine names have no underscore.
#  -DNOUNDERLAPACK    if LAPACK routine names have no underscore.
#  -DBIT64            For I32LP64 systems.
#  -DNOSHORTS         Allow for (LP) blocks of more than 65535 variables.
#  -DUSEOPENMP        To turn on OpenMP parallelization
#  -DSETNUMTHREADS    To use with an OpenMP aware BLAS library.
#
# The default settings for gcc:
#
CFLAGS=-march=core2 -O3 -ansi -Wall -DNOSHORTS -DBIT64 -I/Users/niche/Downloads/Csdp-6.1.0/include 
# AMD Athlon 64 X2:
#  CFLAGS=-march=k8
#
# Notes on CFLAGS.
#
# 1. The -DNOSHORTS flag should always be used.  When this is turned off,
# it causes the code to use unsigned short integers (maximum value 65535)
# in the data structures that describe the problem.  This can cause problems
# when some of the size parameters are larger than 65535.  
#
# 2. By default, we assume that BLAS routines have names like ddot_(),
# but some systems use ddot(), DDOT(), or DDOT_() instead.  A similar issue
# effects the LAPACK routines.  The flags -DCAPSBLAS, -DCAPSLAPACK, 
# -DNOUNDERBLAS. and -DNOUNDERLAPACK are used to handle such situations.
# 
# 3. The code can be built on 64 bit systems that use an I32LP64 model
# in which int's are 32 bits, long's and pointers are 64 bits.  Note that 
# that is the model on all 64 bit Linux and Unix systems that I'm aware of,
# but it is not the model used by MS in its 64 bit Windows!  To build a 
# 64 bit version of the code, use -DBIT64.  You may also need to add a CFLAG
# to tell your compiler to produce 64 bit code.  For example, with gcc, 
# you'll need to add "-m64" to CFLAGS to produce 64 bit code.  
#
# 4. If you have multiple CPU's, and if your compiler supports OpenMP, then
#    you should definitely build a parallel version of CSDP.  To do this,
#    add "-DUSEOPENMP" to CFLAGS.  If your BLAS/LAPACK library routines 
#    use OpenMP's conventions for setting the number of threads to use, then
#    you should also add "-DSETNUMTHREADS".  Note that ATLAS does not 
#    currently work with "-DSETNUMTHREADS", but you can use SETNUMTHREADS
#    on Solaris with -lsunperf and AIX systems with -lesslsmp.

# 5. Using gcc, you can greatly improve the efficency of the code if you 
# specify your processor type.  Examples are given below, use the default 
# if you are unsure.  More examples may be found at 
#
#     http://gcc.gnu.org/onlinedocs/gcc/i386-and-x86_002d64-Options.html
#
# An AMD Athlon XP  based machine:
#  CFLAGS=-march=athlon-xp -O3 -ansi -Wall -DNOSHORTS  -I../include 
# An Intel Pentum 4 based amchine:
#  CFLAGS=-march=pentum4 -O3 -ansi -Wall -DNOSHORTS -I../include 
#
# 6. If you change the CFLAGS, make sure that you use the same CFLAGS
#    in the Makefiles in the solver and theta directories!
#
# The code depends on several libraries:
#
#  sdp                The CSDP subroutine library
#  lapack             The LAPACK linear algebra library
#  blas               The BLAS basic linear algebra library
#
# In addition, if you're using gcc to compile the C code, then you'll need:
#
#  gfortran           The gnu C/Fortran 77 compatibility library
#  m                  The C math run time library.
#  
#
# Use this line to specify where the SDP and linear algebra libraries are
# to be found. 
#
# -L../lib            look in the ../lib directory
# -lsdp               get libsdp.a from the ../lib directory
# -llapack            get liblapack.a or liblapack.so 
# -lblas              get libblas.a or libblas.so
# -lgfortran          get libgfortran.a or libgfortran.so
# -lm                 get libm.a or libm.so
#
# The default settings for a Linux system with gcc3, lapack, blas, and 
# gfortran:
#
LIBS=-lsdp -llapack -lblas -lgfortran -lm
#
# 1. Warning about debian systems.  For some absurb reason, the BLAS
# package in debian has the BLAS library named libblas-3.a, rather than
# libblas.a.  You can either modify the LIBS= line, or you can create
# a symbolic link from /usr/lib/libblas.a to /usr/lib/libblas-3.a.   
#
# 2. An alternative set of flags for use with the ATLAS BLAS on a Linux system.
#
# LIBS=-L../lib -lsdp -llapack -lf77blas -lcblas -latlas -lgfortran -lm 
#
# 3. Special note on gcc 4.  For gcc versions before gcc 4.0, -lgfortran 
#    should be replaced by -lg2c.
#
# 4. An alternative set of flags for use with the ATLAS BLAS, gcc 4.2, and
#    OpenMP on a linux system.
#
# LIBS=-L../lib -lsdp -llapack -lptf77blas -lptcblas -latlas -lgomp -lrt -lpthread -lgfortran -lm
#
#
# Target all builds everything.
#
all: regISM
#
# This builds the mvu code.
#
regISM: regISM.o 
	$(CC) $(CFLAGS) regISM.o $(LIBS) -o regISM
#
# To clean up the directory.
#
clean:
	rm -f *.o
	rm -f regISM
	rm -f prob.dat-s
	rm -f prob.sol
