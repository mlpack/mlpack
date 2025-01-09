## Setting up an mlpack cross-compilation environment

The following table contains a list of different architectures that you can
compile or cross compile mlpack to.  This page assumes that you have a
functional operating system (e.g., Linux, MacOS, Windows) with compatible
ABI on both the host and the target. In addition, your objective is to create
an embedded ABI (eabi). For now, this guide does not support none-eabi or
bare-metal C/C++. The following table refers only to possible values of 
`ARCH_NAME` CMake flag that needs to be specified before the compilation
process starts.

If your device has a fully functional package manager such as apt (for Debian
systems), and you have enough RAM on your embedded device (e.g., RAM > 4GB),
then you will probably be able to compile directly on the device. However, if
this is not the case, then you will need to follow these instructions for
cross-compilation.

Setting up the cross compilation toolchain is generally easy, especially when
using [bootlin toolchains](https://toolchains.bootlin.com/); but other
toolchains work too.  Once you have the toolchain set up, then you need to
identify two parameters to configure mlpack with CMake:

  * `TOOLCHAIN_PREFIX`: this specifies the prefix to use when calling compilers and other tools inside the toolchain
  * `CMAKE_SYSROOT`: this specifies the system root for the cross-compilation environment; in the Bootlin toolchains, this is the `sysroot/` directory
  
For more detailed information on these options see [this tutorial](crosscompile_armv7.md).

You can use the table below with your desired architecture to find links to
appropriate Bootlin toolchains, plus the `TOOLCHAIN_PREFIX` and `CMAKE_SYSROOT`
options you can use with those toolchains.  If you have provided your own
toolchain, you will need to adapt the options accordingly.

Once you have found the correct `TOOLCHAIN_PREFIX` and `CMAKE_SYSROOT` options,
adapt the CMake command below.  ***Don't forget to change
`/path/to/bootlin/toolchain/` to the correct path on your system!***

```sh
cmake \
    -DBUILD_TESTS=ON \
    -DARCH_NAME=(Check the following table) \
    -DCMAKE_CROSSCOMPILING=ON \
    -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=(Check the following table) \
    -DCMAKE_SYSROOT=(Check the following table) \
    ../
```

|-----------------------------------------------------------------------------
| `ARCH_NAME` | link to crosscompiler | CMake command | Example applications |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARM11    | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv6-eabihf.html) | [Sysroot and toolchain prefix](#arm11) | [ARM11 on Wikipedia](https://en.wikipedia.org/wiki/ARM11) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA7 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa7) | [Cortex A7 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A7) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA8 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa8) | [Cortex A8 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A8) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA9  | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa9) |[Cortex A9 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A9) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA15 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa15) | [Cortex A15 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A15)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA53 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html)      | [Sysroot and toolchain prefix](#cortexa53) | [Cortex A53 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A53)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA72 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html)      | [Sysroot and toolchain prefix](#cortexa72) | [Cortex A72 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A72)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORTEXA76 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html)      | [Sysroot and toolchain prefix](#cortexa76) | [Cortex A76 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A76)  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C906      | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_riscv64-lp64d.html)| [Sysroot and toolchain prefix](#c906) | [C906 on riscv](https://www.riscvschool.com/2023/03/09/t-head-xuantie-c906-risc-v/) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x280      | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_riscv64-lp64d.html)| [Sysroot and toolchain prefix](#x280) | [x280 on SiFive](https://www.sifive.cn/api/document-file?uid=x280-datasheet) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| KATAMI    | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-i686.html)     | [Sysroot and toolchain prefix](#katami) |[Pentium 3 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_III)          |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NORTHWOOD | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-64.html)       | [Sysroot and toolchain prefix](#northwood)   |[Pentium 4 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_4)       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| COPPERMINE | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-i686.html)   | [Sysroot and toolchain prefix](#coppermine) |[Pentium 3 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_III)       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

If you didn't see your architecture in the table above, use the closest
architecture with a similar word size, or, adapt the parameters directly in
`CMake/crosscompile-arch-config.cmake` and feel free to open a pull request so we can get
the new architecture added to this table.

### ARM11

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA7 

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA8 

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA9  

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA15 

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA53 

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/aarch64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/aarch64-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA72

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/aarch64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/aarch64-buildroot-linux-gnueabihf/sysroot
```

### CORTEXA76

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/aarch64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/aarch64-buildroot-linux-gnueabihf/sysroot
```

### C906

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/bin/riscv64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/riscv64-buildroot-linux-gnueabihf/sysroot
```

### x280 

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/bin/riscv64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/riscv64-buildroot-linux-gnueabihf/sysroot
```

### KATAMI    

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/bin/x86-i686-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/x86-i686-buildroot-linux-gnueabihf/sysroot
```

### NORTHWOOD

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-64--glibc--stable-2024.02-1/bin/x86-64-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-64--glibc--stable-2024.02-1/x86-64-buildroot-linux-gnueabihf/sysroot
```

### COPPERMINE

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/bin/x86-i686-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/x86-i686-buildroot-linux-gnueabihf/sysroot
```
