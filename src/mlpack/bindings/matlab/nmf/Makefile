nmf: nmf.o
	g++ -O -pthread -shared  \
-Wl,--version-script,/opt/matlab/2010b/extern/lib/glnxa64/mexFunction.map \
-Wl,--no-undefined -o 'mex_nmf.mexa64' nmf.o \
-L../../build/lib -lmlpack \
-Wl,-rpath-link,/opt/matlab/2010b/bin/glnxa64 \
-L/opt/matlab/2010b/bin/glnxa64 -lmx -lmex -lmat -lm -lmwlapack \
-Wl,-rpath=/net/hu19/pmason8/mlpack/trunk/build/lib \
-L/usr/lib64 -larmadillo \

nmf.o:
	g++ -c  \
-I../../build/include \
-I../../build/include/mlpack/methods/nmf \
-I/usr/include/libxml2 \
-I/opt/matlab/2010b/extern/include \
-DMATLAB_MEX_FILE \
-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread \
-DMX_COMPAT_32 -O -DNDEBUG 'nmf.cpp'

clean:
	rm -f *.o *.mexa64
