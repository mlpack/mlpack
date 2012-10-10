emst: emst.o
	g++ -O -pthread -shared  \
-Wl,--version-script,/opt/matlab/2010b/extern/lib/glnxa64/mexFunction.map \
-Wl,--no-undefined -o 'mex_emst.mexa64' emst.o \
-L/net/hu19/pmason8/fastlab/mlpack/trunk/build/lib -lmlpack \
-Wl,-rpath-link,/opt/matlab/2010b/bin/glnxa64 \
-Wl,-rpath=/net/hu19/pmason8/fastlab/mlpack/trunk/build/lib \
-L/opt/matlab/2010b/bin/glnxa64 -lmx -lmex -lmat -lm

emst.o:
	g++ -c  \
-I/net/hu19/pmason8/fastlab/mlpack/trunk/build/include \
-I/net/hu19/pmason8/fastlab/mlpack/trunk/build/include/mlpack/methods/emst \
-I/usr/include/libxml2 \
-I/opt/matlab/2010b/extern/include \
-DMATLAB_MEX_FILE \
-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread \
-DMX_COMPAT_32 -O -DNDEBUG 'emst.cpp'
