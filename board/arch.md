# mlpack on RPI and resource constrained devices

Here is the list of possible architectures that I have in mind,
this list is subject to modifications, new architectures will be added,
and some of these will be deleted.


|  Possible architecture  | Possible toolchains | Known Devices                           |   Priority
|------------------------------------------------------------------------------------------------------
| **ARMv6** (32 bit)      |       GCC           | RPI 0, RPI 1                            |     A
| **ARMv7** (32 bit)      |       GCC           | RPI 2, Beagleboard, Cortex-M devices    |     A
| **ARMv8** (64 bit)      |       GCC           | RPI 2(1.2), RPI 3, RPI 4, Banan Pi,     |     A
| **AVR**                 |       GCC           | Arduino                                 |     C
| **Epiphany**            |       GCC 4.8       | Parallella                              |     A
| **IA-32 (x86)**         |       GCC           | Most Intel from 386+ until 2003         |     B
| **IA-64**(Intel Itanium)|       GCC           | Itanium                                 |     B
| **MIPSI**               |       GCC           | R2000, R3000 (SGI)                      |     B
| **MIPSII**              |       GCC           | R6000 (SGI)                             |     B
| **MIPSIII**             |       GCC           | R4000 (SGI)                             |     B
| **MIPSIV**              |       GCC           | R5000, R8000, R10000 (PlayStation)      |     B
| **MIPSV (32, 64)**      |       GCC           | (Mediatex, Cavium, Nec) VoCore          |     A
| **PA-RISC**             |       GCC           | HP9000 (recent models C8000)            |     C
| **PowerPC**             |       GCC           | (old Apple machine, MacPower            |     C
| **Power ISA V3**        |       GCC           | Power 7,8,9,10                          |     B
| **RISC-V**              |       GCC           | https://www.sifive.com/ HiFive Unleashed|     B
| **SPARCv8**             |       GCC           | TurboSparc                              |     B
| **SPARCv9**             |       GCC           | UltraSparc                              |     B
| **SPARC JPS1**          |       GCC           |                                         |     B
| **SPARC JPS2**          |       GCC           | Sparc64                                 |     B
| **SuperH**              |       GCC           | SH* CPU's used in Sega                  |     D
| **Xtensa**              |       GCC           | ESP* devices,                           |     D
------------------------------------------------------------------------------------------------------

## Dependencies download script
A shell script will download OpenBLAS dependency for mlpack, and cross-compile it.

[comment]:#(

## CMake build system

The idea is to create a Makefile as a wrapper for the actual mlpack CMake
system, this will keep the actual configurations safe and untouched, the wrapper
will be written in the root directory of mlpack containing simple configurations
towards cmake, the following use cases will explain this idea:

### Using Makefile

User will be calling directly make as follows:

```
make mlpack_rpiX
```
Where `X` is the number of raspberry pi  with the following values: `X=0,1,2,3,4`

In this case, make will build only mlpack library for `rpi0`, make will create a
build directory, and call cmake with the arm32 architecture for compilers,
flags and download dependencies.

Since several board have the same architectures, and usually users does not
know the best choice for his/her board, the user will only call `make`
followed by the `board` name, finally make will choose the correct architecture
to build and compile for.

In order to keep compatibility with most `amd64` machines, we can do the
following:

```
make mlpack
```
This will build only mlpack library, to build everything, we can call:

```
make mlpack_all
```
In more generic cases:

```
make mlpack_all_board
```
Also we are interested in several types of devices and not only embedded
systems, processor name can be used in a place of the board, the above table
will give details for these cases and the command to type in order to support
the requested architecture.)
