/**
 * @file build_windows.hpp
 * @author German Lancioni
 * @author Miguel Canteras
 * @author Shikhar Jaiswal
 * @author Ziyang Jiang

@page build_windows Building mlpack From Source on Windows

@section build_windows_intro Introduction

This tutorial will show you how to build mlpack for Windows from source, so
you can later create your own C++ applications, using two different ways:

  - Using CMake to generate an intermeditate Visual Studio solution (`.sln`).
  - @ref build_visual_studio_cmake_integration "Use Visual Studio's CMake integration to directly build from the `CMakeLists`."

Before you try building mlpack, you may
want to install mlpack using vcpkg for Windows. If you don't want to install
using vcpkg, skip this section and continue with the build tutorial.

- Install Git (https://git-scm.com/downloads and execute setup)

- Install CMake (https://cmake.org/ and execute setup)

- Install vcpkg (https://github.com/Microsoft/vcpkg and execute setup)

- To install the mlpack library only:

@code
PS> .\vcpkg install mlpack:x64-windows
@endcode

- To install mlpack and its console programs:
@code
PS> .\vcpkg install mlpack[tools]:x64-windows
@endcode

After installing, in Visual Studio, you can create a new project (or open
an existing one). The library is immediately ready to be included
(via preprocessor directives) and used in your project without additional
configuration.

@section build_windows_env Build Environment

This tutorial has been designed and tested using:
- Windows 10
- Visual Studio 2019 (toolset v142)
- mlpack
- OpenBLAS.0.2.14.1
- boost_1_71_0-msvc-14.2-64
- armadillo (newest version)
- and x64 configuration

The directories and paths used in this tutorial are just for reference purposes.

@section build_windows_prereqs Pre-requisites

- Install CMake for Windows (win64-x64 version from https://cmake.org/download/)
and make sure you can use it from the Command Prompt (may need to add the PATH to 
system environment variables or manually set the PATH before running CMake)

- Download the latest mlpack release from here:
<a href="https://www.mlpack.org/">mlpack website</a>

@section build_windows_instructions Windows build instructions

- Unzip mlpack to "C:\mlpack\mlpack"
- Open Visual Studio and select: File > New > Project from Existing Code
 - Type of project: Visual C++
 - Project location: "C:\mlpack\mlpack"
 - Project name: mlpack
 - Finish
- Make sure the solution configuration is "Debug" and the solution platform is "x64" for this Visual Studio project
- We will use this Visual Studio project to get the OpenBLAS dependency in the next section

@section build_windows_dependencies Dependencies

<b> OpenBLAS Dependency </b>

- Open the NuGet packages manager (Tools > NuGet Package Manager > Manage NuGet Packages for Solution...)
- Click on the “Browse” tab and search for “openblas”
- Click on OpenBlas and check the mlpack project, then click Install
- Once it has finished installing, close Visual Studio

<b> Building OpenBLAS from Source </b>

Unfortunately, the support for building `LAPACK` and `BLAS` on Windows is quite poor, due to the need for Fortran
compiler and libraries. The easiest method to get the necessary `BLAS/LAPACK` libraries built on Windows is to
compile OpenBLAS with LLVM's `clang-cl` and `flang` to produce the required static library (`.lib`) files 
compatible with the MSVC compiler. A comprehensive guide on the
<a href="https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio">compilation
of OpenBLAS for Windows can be found here</a>.

One could always download prebuilt `LAPACK` and `BLAS` libraries for Windows. However, there are few official
sources, and some of those libraries may require further `dll`s at runtime which may not be available in your
system.

It you choose to build `OpenBLAS` from source, make sure that `LAPACK` functions are also built. Finally, make
sure that the `openblas.lib` library is linked in your `Armadillo` build (see below), as well as the library 
path used for the CMake options `BLAS_LIBRARIES` and `LAPACK_LIBRARIES` in the mlpack CMake project.

<b> Boost Dependency </b>

You can either get Boost via NuGet or you can download the prebuilt Windows binaries separately.
This tutorial follows the second approach for simplicity.

- Download the "Prebuilt Windows binaries" of the Boost library ("boost_1_71_0-msvc-14.2-64") from
<a href="https://sourceforge.net/projects/boost/files/boost-binaries/">Sourceforge</a>

@note Make sure you download the MSVC version that matches your Visual Studio

- Install or unzip to "C:\boost\"

<b> Armadillo Dependency </b>

- Download the newest version of Armadillo from <a href="http://arma.sourceforge.net/download.html">Sourceforge</a>
- Unzip to "C:\mlpack\armadillo"
- Create a "build" directory into "C:\mlpack\armadillo\"
- Open the Command Prompt and navigate to "C:\mlpack\armadillo\build"
- Run cmake:

@code
cmake -G "Visual Studio 16 2019" -A x64 -DBLAS_LIBRARY:FILEPATH="C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARY:FILEPATH="C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" ..
@endcode

@note If you are using different directory paths, a different configuration (e.g. Release)
or a different VS version, update the cmake command accordingly. If CMake cannot identify the 
compiler version, check if the Visual Studio compiler and Windows SDK are installed correctly.

- Once it has successfully finished, open "C:\mlpack\armadillo\build\armadillo.sln"
- Build > Build Solution
- Once it has successfully finished, close Visual Studio

@section build_windows_mlpack Building mlpack with CMake-Generated Solution

- Create a "build" directory into "C:\mlpack\mlpack\"
- You can generate the project using either cmake via command line or GUI. If you prefer to use GUI, refer to the \ref build_windows_appendix "appendix"
- To use the CMake command line prompt, open the Command Prompt and navigate to "C:\mlpack\mlpack\build"
- Run cmake:

@code
cmake -G "Visual Studio 16 2019" -A x64 -DBLAS_LIBRARIES:FILEPATH="C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARIES:FILEPATH="C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DARMADILLO_INCLUDE_DIR="C:/mlpack/armadillo/include" -DARMADILLO_LIBRARY:FILEPATH="C:/mlpack/armadillo/build/Debug/armadillo.lib" -DBOOST_INCLUDEDIR:PATH="C:/boost/" -DBOOST_LIBRARYDIR:PATH="C:/boost/lib64-msvc-14.2" -DDEBUG=OFF -DPROFILE=OFF ..
@endcode

@note cmake will attempt to automatically download the ensmallen dependency. If for some reason cmake can't download the dependency, you will need to manually download ensmallen from http://ensmallen.org/ and extract it to "C:\mlpack\mlpack\deps\". Then, specify the path to ensmallen using the flag: -DENSMALLEN_INCLUDE_DIR=C:/mlpack/mlpack/deps/ensmallen/include

- Once CMake configuration has successfully finished, open "C:\mlpack\mlpack\build\mlpack.sln"
- Build > Build Solution (this may be by default in Debug mode)
- Once it has sucessfully finished, you will find the library files you need in: "C:\mlpack\mlpack\build\Debug" (or "C:\mlpack\mlpack\build\Release" if you changed to Release mode)

You are ready to create your first application, take a look at the @ref sample_ml_app "Sample C++ ML App"

@section build_visual_studio_cmake_integration Building mlpack with Visual Studio's CMake Integration

This project can be directly built from the `CMakeLists.txt` with the latest version of MS Visual Studio,
given you have CMake integration via the
<a href="https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-160">C++ 
CMake tools for Windows</a>. To open the CMake project with Visual Studio, select File->Open->CMake
in the top menu, followed by selecting the root `CMakeLists.txt` located in mlpack's root directory.

In order to allow Visual Studio to configure the CMake project, the CMake configuration json will have 
to be edited to provide the <a href="https://github.com/mlpack/mlpack#3-dependencies">relevant options
shown in the `README`</a> needed to find all the dependencies. The options that you
must provide to Visual Studio's CMake are:

  - `ARMADILLO_INCLUDE_DIR`
  - `ARMADILLO_LIBRARY`
  - `BOOST_ROOT`
  - `CEREAL_INCLUDE_DIR`
  - `BLAS_LIBRARIES`
  - `LAPACK_LIBRARIES`

The CMake configuration json can be editted in Visual Studio by right clicking the root `CMakeLists.txt`
in the project view, selecting <b>CMake settings for mlpack</b> and finally clicking on <b>edit JSON</b>.
Adding a new CMake option can be done by adding object fields with the following format to the variables
array in the `CMakeSettings.json`:

@code
{
    "name": "options_name_string",
    "value": "options_value_string",
    "type" : "{BOOL|FILEPATH|PATH|STRING}"
}
@endcode

Here is a full example of the `CMakeSettings.json`file:

@code
{
  "configurations": [
    {
      "name": "x64-Debug (default)",
      "generator": "Ninja",
      "configurationType": "Debug",
      "inheritEnvironments": [ "msvc_x64_x64" ],
      "buildRoot": "${projectDir}\\out\\build\\${name}",
      "installRoot": "${projectDir}\\out\\install\\${name}",
      "cmakeCommandArgs": "",
      "buildCommandArgs": "",
      "ctestCommandArgs": "",
      "variables": [
        {
          "name": "ARMADILLO_INCLUDE_DIR",
          "value": "PATH/TO/CPP/DEPENDENCY/armadillo-10.1.2/include",
          "type": "PATH"
        },
        {
          "name": "ARMADILLO_LIBBRARY",
          "value": "PATH/TO/CPP/DEPENDENCY/armadillo-10.1.2/lib/armadillo.lib",
          "type": "PATH"
        },
        {
          "name": "CEREAL_INCLUDE_DIR",
          "value": "PATH/TO/CPP/DEPENDENCY/cereal-1.3.0/include",
          "type": "PATH"
        },
        {
          "name": "BUILD_ROOT",
          "value": "PATH/TO/CPP/DEPENDENCY/boost_1_66_0",
          "type": "PATH"
        },
        {
          "name": "BOOST_INCLUDEDIR",
          "value": "PATH/TO/CPP/DEPENDENCY/boost_1_66_0",
          "type": "PATH"
        },
        {
          "name": "BLAS_LIBRARIES",
          "value": "PATH/TO/CPP/DEPENDENCY/OpenBLAS/lib/openblas.lib",
          "type": "PATH"
        },
        {
          "name": "LAPACK_LIBRARIES",
          "value": "PATH/TO/CPP/DEPENDENCY/OpenBLAS/lib/openblas.lib",
          "type": "PATH"
        }
      ]
    }
  ]
}
@endcode

@section build_windows_appendix Appendix

If you prefer to use cmake GUI, follow these instructions:

  - To use the CMake GUI, open "CMake".
    - For "Where is the source code:" set `C:\mlpack\mlpack\`
    - For "Where to build the binaries:" set `C:\mlpack\mlpack\build`
    - Click `Configure`
    - If there is an error and Armadillo is not found, try "Add Entry" with the
      following variables and reconfigure:
      - Name: `ARMADILLO_INCLUDE_DIR`; type `PATH`; value `C:/mlpack/armadillo/include/`
      - Name: `ARMADILLO_LIBRARY`; type `FILEPATH`; value `C:/mlpack/armadillo/build/Debug/armadillo.lib`
      - Name: `BLAS_LIBRARY`; type `FILEPATH`; value `C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a`
      - Name: `LAPACK_LIBRARY`; type `FILEPATH`; value `C:/mlpack/mlpack/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a`
    - If there is an error and Boost is not found, try "Add Entry" with the
      following variables and reconfigure:
      - Name: `BOOST_INCLUDEDIR`; type `PATH`; value `C:/boost/`
      - Name: `BOOST_LIBRARYDIR`; type `PATH`; value `C:/boost/lib64-msvc-14.2`
    - Once CMake has configured successfully, hit "Generate" to create the `.sln` file.

@section build_windows_additional_information Additional Information

If you are facing issues during the build process of mlpack, you may take a look at other third-party tutorials for Windows, but they may be out of date:

 * <a href="https://github.com/mlpack/mlpack/wiki/WindowsBuild">Github wiki Windows Build page</a><br/>
 * <a href="http://keon.io/mlpack-on-windows">Keon's tutorial for mlpack 2.0.3</a><br/>
 * <a href="https://overdosedblog.wordpress.com/2016/08/15/once_again/">Kirizaki's tutorial for mlpack 2</a><br/>

*/
