Far more extensive documentation available at
  http://www.mlpack.org/

To build, use CMake:

$ mkdir build
$ cd build
$ cmake ../
$ make
$ sudo make install

If you are suffering segfault, recompile with debugging information:

$ cmake -DDEBUG=ON ../

and you will likely get a listed error as opposed to just a segfault.
