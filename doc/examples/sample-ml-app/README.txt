This directory contains a Visual Studio solution showing the use of mlpack in a
Visual Studio C++ project.

However, you will need to set up your environment correctly first---or modify
the project properties accordingly---in order to build and run the example.

In short, you must download mlpack and four libraries, and install the sources
and library files into C:\mlpack\.

 * OpenBLAS: https://github.com/xianyi/OpenBLAS/releases/download/v0.3.21/OpenBLAS-0.3.21-x64.zip
   Download the .zip, and extract it into C:\mlpack\openblas-0.3.21\

 * Armadillo: https://mlpack.org/files/armadillo-11.4.1.tar.gz
   Download the .tar.gz, and extract it into C:\mlpack\armadillo-11.4.1\; note
   that you may need to use a program such as 7Zip (https://www.7-zip.org/) to
   unpack this archive.

 * Cereal: https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.zip
   Download the .zip, and extract it into C:\mlpack\cereal-1.3.2\

 * ensmallen: https://ensmallen.org/files/ensmallen-2.19.0.tar.gz
   Download the .tar.gz, and extract it into C:\mlpack\ensmallen-2.19.0\

Now, install mlpack into C:\mlpack\mlpack-4.0.0\.  If you downloaded the mlpack
source, you can either use the Windows build guide (see
doc/user/build_windows.md) to build and install, or, since mlpack is
header-only, copy the src/ directory to C:\mlpack\mlpack-4.0.0\ and rename it
"include" (so there will now be a directory C:\mlpack\mlpack-4.0.0\include\,
which contains only base.hpp and the mlpack/ subdirectory).

Alternately, if you downloaded the Windows MSI installer, you can install to
C:\mlpack\mlpack-4.0.0\.

Once all of that setup is done, the example should compile as-is.

If your environment is different, or you have installed dependencies to
different directory, just open the solution Properties and adjust the paths in
Configuration Properties -> C/C++ -> General -> Additional Include Directories,
and Configuration Properties -> Linker -> Input.
