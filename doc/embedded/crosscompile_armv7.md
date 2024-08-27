## Run mlpack bindings on Raspberry PI

In this article, we explore how to run mlpack command-line programs on a
Raspberry Pi 2.  mlpack is a great choice for machine learning tasks on embedded
and low-resource hardware (like the Raspberry Pi) due to its design as a
lightweight header-only C++ library with a focus on efficiency.

<!-- TODO: link to kNN documentation when it's done -->

mlpack provides convenient command-line bindings for many of its algorithms.
In this tutorial, we are going to compile the k-nearest-neighbors command-line
program (kNN), and then run it on a Pi 2.

Since the Pi 2 has 1 GB of RAM, compilation of mlpack code directly on the Pi
can be challenging.  Depending on the OS you are using, 250 MB or more of the
RAM may already be in use just by the system.  This leaves us with around 750 MB
of RAM, and given that the Pi 2 has only ARMv7 core clocked at 1 GHz,
it might take a lot of time for the compilation to finish. As a result it will
be much easier to cross-compile mlpack on a more powerful development system,
and then move the compiled code directly to the Raspberry Pi.

The first step is to download a cross compiler toolchain that is going to run on
an `x86_64` machine (the host) and produce binaries that runs on an ARMv7
machine (this is the Raspberry Pi, the target).  There are several toolchains
around, but the [Bootlin](https://bootlin.com/) toolchains are probably the
easiest to set up and use.  Bootlin has produced toolchains for all kind of
architectures (ARMv7, ARMv8, Sparc, MIPS, RISC-V, etc.), with three different
sets of compilers. To get the toolchain please visit the following website:
[`https://toolchains.bootlin.com/`](https://toolchains.bootlin.com/).

You need to select the architecture and the compiler set. In our case, we are
looking for the arch `armv7-eabihf`, and the libc `glibc`.  You can also
use `musl` or `uclibc` if you are interested in reducing the size of the
compiled programs.  However, in this tutorial, we will stick to glibc for
simplicity.

If you want to get a link directly to the toolchain and download it with `wget`
use the following command:

```sh
wget https://toolchains.bootlin.com/downloads/releases/toolchains/armv7-eabihf/tarballs/armv7-eabihf--glibc--stable-2024.02-1.tar.bz2
```

Once we have downloaded the toolchain we can extract the content using the
following command:

```sh
tar -xvf armv7-eabihf--glibc--stable-2024.02-1.tar.bz2
```

Once we have done that, now the toolchain is ready.  Feel free to explore inside
the toolchain; you will find the basic set of compilers in addition to a
`sysroot/` folder that contains headers for standard libraries.

Now, we will use this toolchain to compile mlpack. The first thing is to get the
mlpack sources.  You can either download the archive file of the latest version,
or clone the repository if you prefer to use the development version.  In this
case we will clone the repository:

```sh
git clone git@github.com:mlpack/mlpack.git
```

Once we have downloaded the source code, now we need to get mlpack external
dependencies.  mlpack has three dependencies:

 * [ensmallen](https://www.ensmallen.org/), which provides mathematical solvers and optimizers for training;
 * [cereal](https://uscilab.github.io/cereal/) for model serialization; and
 * [Armadillo](https://arma.sourceforge.net/) for linear algebra and matrix
   manipulation.

Armadillo itself can use several backends.  In this tutorials we will use
OpenBLAS since it is optimized on a broad range of different architectures.

To get the dependencies we have two solutions: either we download them manually,
or we use the autodownloader.  The three direct dependencies
(ensmallen/cereal/Armadillo) are all header-only and thus it suffices to simply
download and unpack the sources, but Armadillo's backend (the BLAS/LAPACK
implementation) generally is not and so it will need to be cross-compiled for
the target.

In this tutorial we use the autodownloader since it automates the entire
process, including the cross-compilation of OpenBLAS.  The first step is to
create a build directory, just like the
[regular build process](../../README.md#3-installing-and-using-mlpack-in-c):

```sh
cd mlpack/
mkdir build
cd build
```

The next step is to use CMake to configure the build.  In this command we are
telling CMake to use a specific compiler in
addition to pointing out the sysroot that will indicate where the headers and
standard C++ library are.  When cross-compiling, we cannot use the host system
headers or try to link against libraries built for the host.  We specify the
board name to be `RPI2` so that mlpack will configure the compilation options
specifically for the Raspberry Pi 2. Other options for the board name are
available---for instance, `CORTEXA7` could be used too.
You will need to replace `/path/to/bootlin/toolchain/` in the `TOOLCHAIN_PREFIX`
and `CMAKE_SYSROOT` arguments with the location of where you unpacked the
cross-compilation toolchain.

```sh
cmake \
    -DBUILD_TESTS=ON \
    -DBOARD_NAME="RPI2" \
    -DCMAKE_CROSSCOMPILE=ON \
    -DCMAKE_TOOLCHAIN_FILE=../board/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2023.08-1/bin/arm-buildroot-linux-gnueabihf- \
    -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot \
    ../
```

CMake will pull the dependencies and crosscompile OpenBLAS for us.  Once this
has finished, now we just need to call `make` to compile all of the mlpack
command-line programs and tests.  To do this use the following command:

```sh
make -jN
```
Replace `N` with the number of cores you have on your machine.

Alternately, we can compile one method.  In this case, let us try to compile the
k-nearest neighbors (kNN) command-line program using the following command:

```sh
make mlpack_knn
```

Once the above step is completed, we can check the produced binary using the
following command:

```sh
file bin/mlpack_knn
```

This should produce output similar to below, which shows us that we have
compiled a program specifically built to run on the Raspberry Pi:

```
mlpack_knn: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked, for GNU/Linux 3.2.0, stripped
```

Now, we can use the cross-compiled kNN binary. This command will build all the
apps for each one of the algorithms and also build the tests for us. Once this
is done we can transfer all of that to the Pi using `scp`, assuming that the Pi
is available on the network and is running an SSH server.  You will have to
change the path and the IP (e.g. the `192.168.10.100:/home/pi/` part) to match
your setup.

```sh
scp /path/to/mlpack/build/bin/mlpack_knn 192.168.10.100:/home/pi/
```

All of the above apps are statically linked, so they will just run directly on
the Pi without any need to install shared libraries.  Now let us try to run kNN
on the covertype dataset. This dataset is already hosted on mlpack's website. To
pull the dataset run the following commands on the Pi:

```sh
wget http://datasets.mlpack.org/covertype.csv.gz
gzip -d covertype.csv.gz
```

Then to run kNN on the Pi the following command can be used:

```sh
./mlpack_knn \
    --k=5 \
    --reference_file=covertype.csv \
    --distances_file=distances.csv \
    --neighbors_file=neighbors.csv
```

This might take a couple of minutes and around 350 MB of RAM or more.
When finished, it will produce two files.  The first one is `distances.csv`,
which contains the distances between each point in `covertype.csv` and its 5
nearest neighbors.  The second one is `neighbors.csv`, which contains the indices of the 5 nearest neighbors of every point in `covertype.csv`.
If you run into problems, be sure that there is enough RAM for this dataset;
if not, you can reduce the size of the dataset (simply by deleting some rows
from the file) to save some memory.

In the next tutorial we will see how to build cross compilation Makefile to
build one of the examples that we have in the
[mlpack/examples](https://github.com/mlpack/examples) repository.

