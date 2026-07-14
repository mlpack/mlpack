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

| `ARCH_NAME` | link to crosscompiler | CMake command | Example applications |
|-------------|-----------------------|---------------|----------------------|
| ARM11 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv6-eabihf.html) | [Sysroot and toolchain prefix](#arm11) | [ARM11 on Wikipedia](https://en.wikipedia.org/wiki/ARM11) |
| CORTEXA7 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa7) | [Cortex A7 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A7) |
| CORTEXA8 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa8) | [Cortex A8 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A8) |
| CORTEXA9 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa9) | [Cortex A9 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A9) |
| CORTEXA15 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_armv7-eabihf.html) | [Sysroot and toolchain prefix](#cortexa15) | [Cortex A15 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A15) |
| CORTEXA53 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html) | [Sysroot and toolchain prefix](#cortexa53) | [Cortex A53 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A53) |
| CORTEXA72 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html) | [Sysroot and toolchain prefix](#cortexa72) | [Cortex A72 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A72) |
| CORTEXA76 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html) | [Sysroot and toolchain prefix](#cortexa76) | [Cortex A76 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A76) |
| CORTEXA78 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html) | [Sysroot and toolchain prefix](#cortexa78) | [Cortex A78 on Wikipedia](https://en.wikipedia.org/wiki/ARM_Cortex-A78) |
| BCM2711 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_aarch64.html) | [Sysroot and toolchain prefix](#bcm2711) | [Raspberry Pi 4 on Wikipedia](https://en.wikipedia.org/wiki/Raspberry_Pi_4) |
| C906 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_riscv64-lp64d.html) | [Sysroot and toolchain prefix](#c906) | [C906 on riscv](https://www.riscvschool.com/2023/03/09/t-head-xuantie-c906-risc-v/) |
| x280 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_riscv64-lp64d.html) | [Sysroot and toolchain prefix](#x280) | [SiFive x280 product brief](https://www.sifive.com/document-file/sifive-intelligence-x280-product-brief) |
| KATAMI | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-i686.html) | [Sysroot and toolchain prefix](#katami) | [Pentium 3 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_III) |
| NORTHWOOD | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-64.html) | [Sysroot and toolchain prefix](#northwood) | [Pentium 4 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_4) |
| COPPERMINE | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_x86-i686.html) | [Sysroot and toolchain prefix](#coppermine) | [Pentium 3 on Wikipedia](https://en.wikipedia.org/wiki/Pentium_III) |
| POWERPCG4 | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_powerpc-440fp.html) | [Sysroot and toolchain prefix](#powerpcg4) | [Power Mac G4 Cube](https://en.wikipedia.org/wiki/Power_Mac_G4_Cube), [BAE RAD750](https://en.wikipedia.org/wiki/RAD750) |
| MIPS24K | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_mips32.html) | [Sysroot and toolchain prefix](#mips24k) | [MIPS32k on Wikipedia](https://en.wikipedia.org/wiki/MIPS_architecture#MIPS32/MIPS64), [VoCore Ultimate](http://vocore.io/v2u.html) |
| ULTRASPARC | [Bootlin toolchain link](https://toolchains.bootlin.com/releases_sparc64.html) | [Sysroot and toolchain prefix](#ultrasparc) | [UltraSPARC on Wikipedia](https://en.wikipedia.org/wiki/UltraSPARC) |

If you didn't see your architecture in the table above, use the closest
architecture with a similar word size, or, adapt the parameters directly in
`CMake/crosscompile-arch-config.cmake` and feel free to open a pull request so we can get
the new architecture added to this table.

Each section below gives the glibc toolchain by default.  Where Bootlin also
ships a **musl** toolchain for that architecture, a second block is provided:
musl produces a smaller footprint, which can matter on very constrained devices.

### ARM11

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2025.08-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv6-eabihf--glibc--stable-2025.08-1/arm-buildroot-linux-gnueabihf/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_armv6-eabihf.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv6-eabihf--musl--stable-2025.08-1/bin/arm-buildroot-linux-musleabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv6-eabihf--musl--stable-2025.08-1/arm-buildroot-linux-musleabihf/sysroot
```

### CORTEXA7

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/arm-buildroot-linux-gnueabihf/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_armv7-eabihf.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/bin/arm-buildroot-linux-musleabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/arm-buildroot-linux-musleabihf/sysroot
```

### CORTEXA8

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/arm-buildroot-linux-gnueabihf/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_armv7-eabihf.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/bin/arm-buildroot-linux-musleabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/arm-buildroot-linux-musleabihf/sysroot
```

### CORTEXA9

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/arm-buildroot-linux-gnueabihf/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_armv7-eabihf.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/bin/arm-buildroot-linux-musleabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/arm-buildroot-linux-musleabihf/sysroot
```

### CORTEXA15

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/bin/arm-buildroot-linux-gnueabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--glibc--stable-2025.08-1/arm-buildroot-linux-gnueabihf/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_armv7-eabihf.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/bin/arm-buildroot-linux-musleabihf-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/armv7-eabihf--musl--stable-2025.08-1/arm-buildroot-linux-musleabihf/sysroot
```

### CORTEXA53

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/bin/aarch64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_aarch64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/bin/aarch64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/aarch64-buildroot-linux-musl/sysroot
```

### CORTEXA72

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/bin/aarch64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_aarch64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/bin/aarch64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/aarch64-buildroot-linux-musl/sysroot
```

### CORTEXA76

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/bin/aarch64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_aarch64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/bin/aarch64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/aarch64-buildroot-linux-musl/sysroot
```

### CORTEXA78

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/bin/aarch64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_aarch64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/bin/aarch64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/aarch64-buildroot-linux-musl/sysroot
```

### BCM2711

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/bin/aarch64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_aarch64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/bin/aarch64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/aarch64--musl--stable-2025.08-1/aarch64-buildroot-linux-musl/sysroot
```

### C906

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2025.08-1/bin/riscv64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2025.08-1/riscv64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_riscv64-lp64d.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--musl--stable-2025.08-1/bin/riscv64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--musl--stable-2025.08-1/riscv64-buildroot-linux-musl/sysroot
```

### x280

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2025.08-1/bin/riscv64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--glibc--stable-2025.08-1/riscv64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_riscv64-lp64d.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/riscv64-lp64d--musl--stable-2025.08-1/bin/riscv64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/riscv64-lp64d--musl--stable-2025.08-1/riscv64-buildroot-linux-musl/sysroot
```

### KATAMI

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2025.08-1/bin/i686-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2025.08-1/i686-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_x86-i686.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--musl--stable-2025.08-1/bin/i686-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--musl--stable-2025.08-1/i686-buildroot-linux-musl/sysroot
```

### NORTHWOOD

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-64--glibc--stable-2025.08-1/bin/x86_64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-64--glibc--stable-2025.08-1/x86_64-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_x86-64.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-64--musl--stable-2025.08-1/bin/x86_64-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-64--musl--stable-2025.08-1/x86_64-buildroot-linux-musl/sysroot
```

### COPPERMINE

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2025.08-1/bin/i686-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--glibc--stable-2025.08-1/i686-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_x86-i686.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/x86-i686--musl--stable-2025.08-1/bin/i686-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/x86-i686--musl--stable-2025.08-1/i686-buildroot-linux-musl/sysroot
```

### POWERPCG4

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/powerpc-440fp--glibc--stable-2025.08-1/bin/powerpc-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/powerpc-440fp--glibc--stable-2025.08-1/powerpc-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_powerpc-440fp.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/powerpc-440fp--musl--stable-2025.08-1/bin/powerpc-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/powerpc-440fp--musl--stable-2025.08-1/powerpc-buildroot-linux-musl/sysroot
```

### MIPS24K

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/mips32--glibc--stable-2025.08-1/bin/mips-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/mips32--glibc--stable-2025.08-1/mips-buildroot-linux-gnu/sysroot
```

For a smaller footprint, Bootlin also provides a
[musl libc toolchain](https://toolchains.bootlin.com/releases_mips32.html);
use these variables instead:

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/mips32--musl--stable-2025.08-1/bin/mips-buildroot-linux-musl-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/mips32--musl--stable-2025.08-1/mips-buildroot-linux-musl/sysroot
```

### ULTRASPARC

Bootlin ships sparc64 as a glibc toolchain only (no musl variant is available).

```
-DTOOLCHAIN_PREFIX=/path/to/bootlin/toolchain/sparc64--glibc--stable-2025.08-1/bin/sparc64-buildroot-linux-gnu-
-DCMAKE_SYSROOT=/path/to/bootlin/toolchain/sparc64--glibc--stable-2025.08-1/sparc64-buildroot-linux-gnu/sysroot
```

***Note:*** the sparc64 instruction set does not support unaligned loads;
therefore, if [image operations](../user/load_save.md#image-data) are being
performed, add `#define STBIR_MEMCPY_NOUNALIGNED` to your code before including
mlpack, or add the compiler option `-DSTBIR_MEMCPY_NOUNALIGNED`.
