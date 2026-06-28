## On-device movement recognition with an IMU

In this tutorial we build a complete, end-to-end **human-movement / activity
recognition** pipeline that runs *entirely on a tiny RISC-V Linux board*. The
pipeline consists of data collection, training, and inference all happen on the target device.
The main target of this tutorial is to provide an end to end example of using mlpack
on real resource-constrained embedded hardware.

If you have not cross-compiled mlpack before, read these first:

 * [Run mlpack bindings on a Raspberry Pi](crosscompile_armv7.md)
 * [Cross-compile an mlpack example for embedded hardware](crosscompile_example.md)
 * [Cross-compilation setup (toolchains per board)](supported_boards.md)

The full source code for this tutorial can be found in the
[mlpack examples repository](https://github.com/mlpack/examples), under
[`cpp/movement_recognition/`](https://github.com/mlpack/examples/tree/master/cpp/movement_recognition).

Contents:

 * [What are we building](#what-are-we-building)
 * [Hardware](#hardware)
 * [Setting up the cross-compilation toolchain](#setting-up-the-cross-compilation-toolchain)
 * [Getting the example](#getting-the-example)
 * [Building the data-collection tools](#building-the-data-collection-tools)
 * [Building the training and inference programs](#building-the-training-and-inference-programs)
 * [Copying everything to the device](#copying-everything-to-the-device)
 * [Running it on the device](#running-it-on-the-device)
 * [Annex A: shrinking the binary (image and audio support)](#annex-a-shrinking-the-binary-image-and-audio-support)
 * [Annex B: cross-compiling ncurses for the TUI](#annex-b-cross-compiling-ncurses-for-the-tui)
 * [Annex C: making OpenBLAS fit (so training runs on the device)](#annex-c-making-openblas-fit-so-training-runs-on-the-device)

### What are we building

We are building a machine learning based human movement recognition to detect
movements such as  *walking*, *sitting*, *squats*, and *climbing stairs*. This
is enabled by using a 9 Degree of Freedom inertial sensor that is read over an I2C bus.
The collected data is cut into windows, and then fed into a Fast Fourier
Transform in order to extract the features from each collected windows.
Finally, we build a small `float32` neural network learns to recognize the
movements, with the highest possible accuracy. 

The example is split into four small independent programs. Each one of them
can be run by the user to do one specific task. It is up to the user to
combine them or integrate such a functionality in their project. Our main
target is to provide an example on how such a software can be built: 

| Program          | Built from         | Uses mlpack? | Purpose |
|------------------|--------------------|:------------:|---------|
| `driver/imu_test`| `driver/`          | no           | sensor check + magnetometer calibration |
| `collect`        | `collect/`         | no           | record sensor data to CSV (small, exception-free) |
| `train`          | `train/train.cpp`  | yes          | train a neural network from the CSVs |
| `infer`          | `infer/infer.cpp`  | yes          | live inference from the IMU |

`collect` and `imu_test` only need the Linux I2C headers, so they compile to
~150–170 KB static binaries with their own `Makefile`s.  `train` and `infer`
link mlpack and are built with CMake using the same cross-compilation
infrastructure described in the [embedded example tutorial](crosscompile_example.md).

The pipeline is: **`collect`** writes one CSV file per recording (the file name
is the label), **`train`** turns those CSVs into FFT features and fits a network,
and **`infer`** reads the live sensor stream and prints the predicted movement.

### Hardware

This tutorial uses a **[Milk-V Duo](https://milkv.io/duo)** — a SOPHGO
CV1800B board with a dual-core RISC-V C906 CPU and **64 MB of RAM** (of which
only ~28 MB is usable from Linux), running a musl-based Linux.  The sensor is a
**GY-89** 10-DOF breakout, which carries three separate I2C chips:

| Chip        | Function                            | 7-bit address |
|-------------|-------------------------------------|---------------|
| **L3GD20H** | 3-axis gyroscope                    | `0x6B`        |
| **LSM303D** | 3-axis accelerometer + magnetometer | `0x1D`        |
| **BMP180**  | barometric pressure + temperature   | `0x77`        |

Wire the GY-89 to the Duo's I²C0 bus (the example defaults to `/dev/i2c-0`):

| GY-89 pin | Milk-V Duo |
|-----------|------------|
| VIN/VCC   | 3V3        |
| GND       | GND        |
| SCL       | IIC0_SCL (GP0) |
| SDA       | IIC0_SDA (GP1) |

On the Duo those pads / pins default to a different function and must be muxed to I2C.
This is done on the device with `duo-pinmux` and is shown in
[Running it on the device](#running-it-on-the-device). Feel free to check
pinmux software on the Duo to change the functionality of the pins and use a
different interface.

### Setting up the cross-compilation toolchain

Since the device is resource constrained with only 28 MB available
RAM, therefore, we cross-compile on a host `x86_64` machine and copy the static
binaries on the target machine, exactly as we did in the [Raspberry Pi tutorial](crosscompile_armv7.md).
The board uses a RISC-V C906 core, so we need a `riscv64-lp64d`
[Bootlin](https://toolchains.bootlin.com/) toolchain.  Our target in this tutorial to produce
the smallest static binary we use the **musl** variant

```sh
wget https://toolchains.bootlin.com/downloads/releases/toolchains/riscv64-lp64d/tarballs/riscv64-lp64d--musl--stable-2024.02-1.tar.bz2
tar -xvf riscv64-lp64d--musl--stable-2024.02-1.tar.bz2
```

For the rest of the tutorial we refer to the unpacked toolchain through two
shell variables; adjust the path to where you extracted it:

```sh
export TC=/path/to/riscv64-lp64d--musl--stable-2024.02-1
export GXX=$TC/bin/riscv64-buildroot-linux-musl-g++
```

The C906 architecture and its toolchain prefix/sysroot are listed on the
[cross-compilation setup page](supported_boards.md#c906). However, note that
we are using the musl toolchain here instead of the glibc one shown there.

### Getting the example

Clone the examples repository and move into the example directory:

```sh
git clone https://github.com/mlpack/examples.git
cd examples/cpp/movement_recognition
```

### Building the data-collection tools

`imu_test` (sensor check + magnetometer calibration) and `collect` (recording)
have no mlpack dependency, so they build directly with their own `Makefile`s and
the cross toolchain.  Pass the cross compiler and ask for a
static binary:

```sh
# Sensor check + magnetometer calibration  ->  driver/imu_test
make -C driver CXX=$GXX STATIC=-static
```

`collect` includes an interactive ncurses front-end by default (pass `--no-tui` at runtime
for plain command-line recording). ncurses is not part of the toolchain, so point the build
at a target ncurses you cross-build once (see [Annex B](#annex-b-cross-compiling-ncurses-for-the-tui)):

```sh
export P=/path/to/ncurses-riscv64

# Data-collection tool  ->  collect/collect
make -C collect CXX=$GXX STATIC=-static \
     NCURSES_CFLAGS="-isystem $P/include -isystem $P/include/ncursesw" \
     NCURSES_LIBS="-L$P/lib -lncursesw"
```

If you do not need the TUI, you can build a smaller command line only `collect` with
`make -C collect CXX=$GXX STATIC=-static NCURSES=0` (then always pass
`--no-tui`).

### Building the training and inference programs

`train` and `infer` link against mlpack, so they use the CMake flow, which reuses the
repository's embedded cross-compile machinery (`CMake/`) to fetch mlpack and its
dependencies and cross-compile OpenBLAS, as described in 
[embedded example tutorial](crosscompile_example.md). 
At this stage, we need to define the architecture of the target device, using
the following variable and the riscv-cpu as follows:`ARCH_NAME=C906`:

```sh
mkdir build && cd build
cmake \
    -DCMAKE_CROSSCOMPILING=ON \
    -DARCH_NAME=C906 \
    -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=$TC/bin/riscv64-buildroot-linux-musl- \
    -DCMAKE_SYSROOT=$TC/riscv64-buildroot-linux-musl/sysroot \
    -DNCURSES_PREFIX=$P \
    ..
make train infer
```

`ARCH_NAME=C906` selects C906 tuning (`-mtune=thead-c906`) and a scalar
`RISCV64_GENERIC` OpenBLAS build (no large vector buffers, friendlier on
64 MB), and OpenMP is disabled (the board is effectively single-core for this
workload), check [Annex C](#annex-c-making-openblas-fit-so-training-runs-on-the-device),
for more details regarding OpenBLAS optimization for this specific device.
`NCURSES_PREFIX` points `infer` at the same target ncurses used for
`collect`; if it is omitted, `infer` is skipped and only `train` is built.

When the build finishes you will have `driver/imu_test`, `collect/collect`, and
`build/train` + `build/infer`, all static RISC-V binaries:

```sh
file build/train
# build/train: ELF 64-bit LSB executable, UCB RISC-V, ... statically linked, stripped
```

### Copying everything to the device

The Duo's BusyBox userland has no SFTP server, so a plain `scp` fails with
`sh: /usr/libexec/sftp-server: not found`.  Use scp's **legacy protocol** with
`-O`.  Over the Duo's USB-OTG connection the board is usually reachable at
`192.168.42.1`, you can check that by doing a local ping. The default password
for the Duo is `milkv`:

```sh
scp -O driver/imu_test  root@192.168.42.1:/root/imu_test
scp -O collect/collect  root@192.168.42.1:/root/collect
scp -O build/train      root@192.168.42.1:/root/train
scp -O build/infer      root@192.168.42.1:/root/infer
ssh root@192.168.42.1 'chmod +x /root/imu_test /root/collect /root/train /root/infer'
```

### Running it on the device

SSH into the board.  All the commands below run **on the Duo**.

**1. Check the I2C0 pins and the sensor.** The GP0/GP1 pads must be set to
their I2C function first:

```sh
i2cdetect -y -r 0          # should show devices at 0x1d and 0x6b (and 0x77)
```

This is the step most likely to be missing if the sensor does not respond.
Therefore, you can mux the pins and change their functionality as follows:

```sh
duo-pinmux -p GP0 -f IIC0_SCL
duo-pinmux -p GP1 -f IIC0_SDA
i2cdetect -y -r 0          # Now it should show devices at 0x1d and 0x6b (and 0x77)
```

**2. Check the sensor and calibrate the magnetometer**
Even though we are going to use Accelerometer only. Calibration is important and it is 
built into `imu_test`; rotate the board through all orientations while it samples:

```sh
./imu_test --calibrate mag.cal /dev/i2c-0 20
```

**3. Collect labelled data.**  Each recording is written to its own file named
`<label>_<date>.csv`, so the *label is the file name*.  Run `collect` once per
movement; the interactive TUI is the default (pick the label, toggle sensors,
press **Space** to start/stop), or use `--no-tui` for scripted capture over SSH:

```sh
mkdir data
./collect --no-tui --label walking   --sensors accel --duration 30 --out-dir data
./collect --no-tui --label sitting   --sensors accel --duration 30 --out-dir data
./collect --no-tui --label squat     --sensors accel --duration 30 --out-dir data
```

Collect several recordings per movement (more files means more training
windows), keeping the same `--sensors` selection across all of them.

**4. Train the network.**  `train` groups the CSVs by label, cuts each into
non-overlapping windows of `--window` samples, runs one FFT per channel to get
features, and trains a small `float32` neural network.  Instead of a fixed epoch
count it uses **early stopping**: `--patience` is how many epochs it keeps
searching after the lowest validation loss before stopping:

```sh
./train --data data --window 64 --patience 10 --out model
```

It prints a per-epoch loss and a progress bar while training, then a held-out
test accuracy, and writes `model.bin` (the trained weights) and `model.labels`
(window size + class names).

**5. Run live inference.**  `infer` reads the IMU, slides the same window over
the stream, runs the same FFT, and prints the predicted movement.  Pass the same
`--sensors` you trained with:

```sh
./infer --bundle model --sensors accel            # prints predictions to stdout
./infer --bundle model --sensors accel --tui      # interactive ncurses view
```

You can pass `--bundle` more than once to compare several trained networks on
the same live stream.

### Annex A: shrinking the binary (image and audio support)

When you `#include <mlpack.hpp>`, mlpack's `data::Load`/`data::Save` pull in
support for **image** files (via the bundled STB libraries) and **audio** files
(via the bundled dr_libs `dr_mp3`/`dr_wav`).  In this tutorial, we only loads CSVs,
However, the image and audio function symbols is already part of the binary. It
would be great to remove this dead code by disabling these two libraries to
reduce the footprint of the static binary. Removing these two will allow us to
reduce approximately 100 KB from the target binary.

mlpack provides [compile-time options](../user/compile.md#configuring-mlpack-with-compile-time-definitions)
that can compile both out as follows:

| CMake option | Code define | Effect |
|--------------|-------------|--------|
| `-DDISABLE_STB=ON`     | `MLPACK_DISABLE_STB`     | Remove image (STB) support |
| `-DDISABLE_DR_LIBS=ON` | `MLPACK_DISABLE_DR_LIBS` | Remove audio (dr_libs) support |


Since `train`/`infer` have their own CMake files, and they are using a local
none installed headers of mlpack. Therefore We can add these compile definition
directly from cmake as follows:

```cmake
add_compile_definitions(MLPACK_DISABLE_STB)
# add_compile_definitions(MLPACK_DISABLE_DR_LIBS)  # if your mlpack provides it
```

By far the largest single saving in this example, however, is **not** a
third-party switch: serializing a neural network with
`MLPACK_ENABLE_ANN_SERIALIZATION` registers *every* layer type and adds well
over a megabyte.  `train` avoids it entirely by saving only the trained weight
matrix (`net.Parameters()`) and rebuilding the fixed architecture in `infer`.

### Annex B: cross-compiling ncurses for the TUI

The `collect --tui` and `infer --tui` front-ends need ncurses, which the Bootlin
toolchain does not ship for the target.  Cross-build it once (static, with a few
terminal fallbacks baked in so no terminfo database is needed on the device),
then point the two builds at it with `NCURSES_PREFIX` (and the
`NCURSES_CFLAGS`/`NCURSES_LIBS` variables for `collect`):

```sh
# In an unpacked ncurses source tree; P is where it gets installed:
export P=/path/to/ncurses-riscv64
./configure --host=riscv64-buildroot-linux-musl CC=$TC/bin/riscv64-buildroot-linux-musl-gcc \
    --prefix=$P --without-shared --without-cxx --without-progs \
    --disable-database --with-build-cc=gcc \
    --with-fallbacks=linux,vt100,xterm,xterm-256color,screen,ansi
make && make install
```

After this, `$P/lib/libncursesw.a` and the headers under `$P/include` are what
the `collect` and `train`/`infer` build commands above expect.  If you do not
need the interactive front-ends, build `collect` with `NCURSES=0` and omit
`-DNCURSES_PREFIX`, and use the command-line modes (`collect --no-tui`,
`infer` without `--tui`).

### Annex C: making OpenBLAS fit (so training runs on the device)

This is the single most important thing to understand for training *on* a
64 MB board, so it is worth reading carefully even though the example already
does it for you.

In the following we are detailing necessary modification relevant to OpenBLAS
to make it able to run neural network on resource constrained device. The
current default configuration, the neural network training step would freeze when 
we try to run it on the target device. The raison for this is the matrix
multiplication function (GEMM). By default GEMM allocates an internal buffer
with default `BUFFER_SIZE=32` MB allocated, which we are going to reduce to 8 MB.

In addition to this, we need to reduce the number of columns block treated by
the CPU at one time, since the cache is lower on our device. We need to reduce
this from `SGEMM_DEFAULT_R = 12288` to `2048`.

The above reduction is possible for two reason: the first one is that this 
board has only 28 MB available, and second reason is that all also our matrices
are tiny in this example (e.g., 64 x 297). 

The fix is simple, we patch two headers in OpenBLAS before it is compiled in 
`CMake/ConfigureCrossCompile.cmake` as follows:

```cmake
# NN-on-device memory fit (riscv64).  OpenBLAS lazily allocates a per-GEMM
# scratch buffer (BUFFER_SIZE -- 32 MB on riscv64) sized for its default
# N-block (SGEMM_DEFAULT_R = 12288).  That single 32 MB allocation does
# not fit on a ~28 MB device, so the first f32 matrix-multiply -- e.g. the
# neural network's dense layers -- is OOM-killed at startup.  
# Our matrices are tiny, so shrink the N-block to 2048 and the buffer to 8 MB.
if(OPENBLAS_TARGET STREQUAL "RISCV64_GENERIC")
  execute_process(COMMAND sed -i
      "/#ifdef RISCV64_GENERIC/,/#endif/{s/_DEFAULT_R 12288/_DEFAULT_R 2048/;s/_DEFAULT_R 8192/_DEFAULT_R 2048/;s/_DEFAULT_R 4096/_DEFAULT_R 2048/}"
      "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/param.h")
  execute_process(COMMAND sed -i
      "s/( 32 << 20)/( 8 << 20)/"
      "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/common_riscv64.h")
endif()
```

In addition to this, OpenBLAS is build as a single thread, since the cpu is a
single-core:

`USE_THREAD=0 NUM_THREADS=1 USE_OPENMP=0`:

```cmake
execute_process(COMMAND make TARGET=${OPENBLAS_TARGET} BINARY=${OPENBLAS_BINARY}
    HOSTCC=gcc CC=${CMAKE_C_COMPILER} FC=${CMAKE_FORTRAN_COMPILER}
    NO_SHARED=1 USE_THREAD=0 NUM_THREADS=1 USE_OPENMP=0
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version})
```
