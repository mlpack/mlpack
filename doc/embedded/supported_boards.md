## List of supported architectures:

The following table contains a list of different architectures that you can
compile or cross compile mlpack to, this page assumes that you have a
functional operating system (e.g., Linux, MacOS, Windows) with compatible
ABI on both the host and the target. In addition, your objective is to create
an embedded ABI (eabi). For now, this guide does not support none-eabi or
bare-metal C/C++. The following table refers only to possible values of 
`BOARD_NAME` CMake flag that needs to be specified before the compilation
process starts.

please check its architecture of your board before specifying the parameter, if
your architecture is not part of the board please refer to the closest
architecture with similar word size, or adapt the parameters and feel free to
open a pull request.

If your device has a fully functional package manager such as debian, and you
have enough RAM on your embedded device (e.g., RAM > 4GB), then you will
probably be able to compile directly on the device. However, if these are not
available then you will need to follow these instruction to do cross
compilation.

Setting up the cross compilation toolchain is basically easy. We usually use
bootlin [toolchains](https://toolchains.bootlin.com/) but this is not
obligatory. If you have a specific architecture, then you need you need to
identify two parameters that you need fill them in the first one is the
`TOOLCHAIN_PREFIX` and the second one is the `DCMAKE_SYSROOT`. For more
detailed example please check this [tutorial](crosscompile_armv7.md)

Here is the generic CMake command, please check the table to modify it
accoding to each architecture and the version of your compiler set, also
do not forget to change `/path/to/bootlin/toolchain` by the real path:

```sh
cmake \
    -DBUILD_TESTS=ON \
    -DBOARD_NAME="ARM11" \
    -DCMAKE_CROSSCOMPILE=ON \
    -DCMAKE_TOOLCHAIN_FILE=../board/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=(Check the following table) \
    -DCMAKE_SYSROOT=(Check the following table) \
    ../
```

| Architecture\_NAME | link to crosscompiler | CMake command |  
|------------------------------------------------------------|
| ARM11    | [compiler set](https://toolchains.bootlin.com/releases_armv6-eabihf.html) | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2024.05-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2024.05-1/arm-buildroot-linux-gnueabihf/sysroot`  | 
|-------------------------------------------------------------------------------------------------|
| CORTEXA7 | [compiler set](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|------------------------------------------------------------------------------------------------|
| CORTEXA8 | [compiler set](https://toolchains.bootlin.com/releases_armv7-eabihf.html)  | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot`|
|---------------------------------------------------------------------------------------|
| CORTEXA9  | [compiler set](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| CORTEXA15 | [compiler set](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | ` -DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot`
|---------------------------------------------------------------------------------------|
| CORTEXA53 | [compiler set](https://toolchains.bootlin.com/releases_aarch64.html)      | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| CORTEXA72 | [compiler set](https://toolchains.bootlin.com/releases_aarch64.html)      | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| CORTEXA76 | [compiler set](https://toolchains.bootlin.com/releases_aarch64.html)      | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| C906      | [compiler set](https://toolchains.bootlin.com/releases_riscv64-lp64d.html)| `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| x280      | [compiler set](https://toolchains.bootlin.com/releases_riscv64-lp64d.html)| `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|
| KATAMI    | [compiler set](https://toolchains.bootlin.com/releases_x86-i686.html)     | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot`|
|---------------------------------------------------------------------------------------|
| NORTHWOOD | [compiler set](https://toolchains.bootlin.com/releases_x86-64.html)       | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-64--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-64--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot`   |
|---------------------------------------------------------------------------------------|
| COPPERMINE |  [compiler set](https://toolchains.bootlin.com/releases_x86-i686.html)   | `-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/bin/arm-buildroot-linux-gnueabihf- -DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2024.02-1/arm-buildroot-linux-gnueabihf/sysroot` |
|---------------------------------------------------------------------------------------|

