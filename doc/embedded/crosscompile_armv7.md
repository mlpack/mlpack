## Run mlpack bindings on Raspberry PI

In this article we are going to explore how to run mlpack command line programs on a
Raspsberry PI, if you do not know mlpack, it is a header-only C++ machine
learning library that is desinged for efficiency. mlpack contains more than 40
machine learning algorithms includuing neural networks, and binding to other
launguages such as Python, R or Julia.

Each one of mlpack algorithms has a small app that allows you to run the
algorithm from the command line. In today's tutorials we are going to compile
these apps, so we can run a couple of them from the command line on a Raspberry
PI.

Since the RPI 2 has 1 GB of RAM, depending on the OS you are using you might find
at least 250 MB of these are already used by default. This leaves us with around
750 MB of RAM, this is not enough to compile all the apps that we already have,
and given the fact that we only dispose of 1 GhZ ARMv7 core,
it might take a lot of time for the compilation to finish. As a result it will be easier
to cross-compile all mlpack apps on your hadware including the integration tests, and then
move all them directly to the RPI.

The first step is to download a cross compiler toolchain that is going to run on an
x86\_64 machine referred to as the host and produce binaries that runs on an ARMv7
referred to as the target. There are several toolchains around, but my favourait
are the ones produced by Bootlin, they have
produced toolchains for all kind of architectures, ARMv7, ARMv8, Sparc, MIPS,
RISC-V, etc, with three different set of compilers. To get the toolchain please
visit the following website: https://toolchains.bootlin.com/

You need to select the architecture and the compiler set. In our case, we are
looking for armv7-eabihf, and we will use glibc in this tutorials, you can also
use musl or uclibc if you are interested in an optimized apps footprints.
However, in this tutorials, we will stick to glibc for simplicity.

If you want to get a link directly to the toolchain and download it with wget
use the following command:

```
wget https://toolchains.bootlin.com/downloads/releases/toolchains/armv7-eabihf/tarballs/armv7-eabihf--glibc--stable-2024.02-1.tar.bz2
```

Once we have downloaded the toolchain we can extract the content using the
following command:

```
tar -xvf armv7-eabihf--glibc--stable-2024.02-1.tar.bz2
```

Once we have done that, now the toolchain is ready, feel free to explore inside
the toolchain you will find the basic set of compilers in addition to a sysroot
folder that contains standard libraries headers.

Now, we aim to use this toolchain to compile mlpack. The first thing we need to
do is to get the source code of the library, either you can download the zip
file of the last version on github or basically clone the main branch. In this
case I will clone the main branch

```
git clone git@github.com:mlpack/mlpack.git
```

Once we have downloaded the source code, now we need to get mlpack external
dependencies. mlpack has three dependcies, ensmallen which
provides mathematical solvers and optimisers for training, Cereal for model
serialization, and armadillo for linear algebra and matrix manipulation.
Armadillo itself can use several backends, in this tutorials we will use
OpenBLAS since it is optimized on a broad range of different architectures.


To get the dependencies we have two solutions, either we download them manually
or we use the autodownloader. In this I will use the autdownloader since
it automate the entire process, it pull the dependencies and start the cross
compilation process at once.

```
cd mlpack/
mkdir build
cd build
```

In this command we are basically telling cmake to use this specific compiler in
addition to pointing out the sysroot that will indicate where are the headers
and the standard C++ library. In cross compilation we should not use system
headers or try to link against the host library. We are specifying the board
name to be `RPI2` also this can be replaced with `CORTEXA7`. You will need to specify
the path for the toolchain and for the mlpack library.

```
cmake -DBUILD_TESTS=ON -DBOARD_NAME="RPI2" -DCMAKE_CROSSCOMPILE=ON -DCMAKE_TOOLCHAIN_FILE=/path/to/mlpack/board/crosscompile-toolchain.cmake -DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2023.08-1/bin/arm-buildroot-linux-gnueabihf-  -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot  ../
```

CMake will pull the dependencies and crosscompile openblas for us, once this
has finished now we need to call make to compile mlpack CLI binaries. To compile
all CLI method use the following command

```
make -jN
```
Replace `N` with the number of cores you have on your machine.

Otherwise, we can compile one method, in this case, let us try to compile k-nearest neighbors
(kNN) using the following command:

```
make mlpack_knn
```

Once the above step is completed, we can check the produced binary using the
following command:

```
file bin/mlpack_knn
mlpack_knn: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked, for GNU/Linux 3.2.0, stripped
```

Once we have finished we can check the produced kNN binary 
This command will build all the apps for each one of the algorithms and also
build the tests for us. Once this is done we can transfer all of that to the
PI using scp, please change the path to and the IP address as it should be.

```
scp -r /path/to/mlpack/build/bin/mlpack_knn 192.168.10.100:/home/pi/
```

All of the above apps are statically linked, they will just run on the PI, let
us try to run kNN on the covertype dataset. This dataset is already hosted on
mlpack dataset website. To pull the dataset use the following command:

```
wget http://datasets.mlpack.org/covertype.csv.gz

gzip -d covertype.csv.gz
```

Then to run it on the PI from the terminal using the following command:
```
./mlpack_knn --k=5 --reference_file=covertype.csv --distances_file=distances.csv --neighbors_file=neighbors.csv
```

This might take a couple of minutes and around 350 MB of RAM or more,
eventually it will produce two files, the first one is the distances to the
neighbors and the second one contains the index to the neighbors in the
original dataset. Also be sure that there is enough RAM for this dataset.
Also you can reduce the size of the dataset to save some memory.

In the next tutorial we will see how to build cross compilation Makefile to
build one of the examples that we have in the mlpack/examples repository.

