range_search: range_search.o
	g++ -O -pthread -shared  \
-Wl,--version-script,/opt/matlab/2010b/extern/lib/glnxa64/mexFunction.map \
-Wl,--no-undefined -o 'mex_range_search.mexa64' range_search.o \
-L../../build/lib -lmlpack \
-Wl,-rpath-link,/opt/matlab/2010b/bin/glnxa64 \
-L/opt/matlab/2010b/bin/glnxa64 -lmx -lmex -lmat -lm \
-Wl,-rpath=/net/hu19/pmason8/mlpack/trunk/build/lib \

range_search.o:
	g++ -c  \
-I../../build/include \
-I../../build/include/mlpack/methods/range_search \
-I/usr/include/libxml2 \
-I/opt/matlab/2010b/extern/include \
-DMATLAB_MEX_FILE \
-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread \
-DMX_COMPAT_32 -O -DNDEBUG 'range_search.cpp'

clean:
	rm -f *.o *.mexa64
