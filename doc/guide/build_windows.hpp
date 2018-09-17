/**
 * @file build_windows.hpp
 * @author German Lancioni

@page build_windows Building mlpack From Source on Windows

@section build_windows_intro Introduction

This document discusses how to build mlpack for Windows from source, so you can
later create your own C++ applications.  There are a couple of other tutorials
for Windows, but they may be out of date:

 * <a href="https://github.com/mlpack/mlpack/wiki/WindowsBuild">Github wiki Windows Build page</a>
 * <a href="http://keon.io/mlpack-on-windows">Keon's tutorial for mlpack 2.0.3</a>
 * <a href="https://overdosedblog.wordpress.com/2016/08/15/once_again/">Kirizaki's tutorial for mlpack 2</a>

Those guides could be used in addition to this tutorial.

@section build_windows_env Environment

This tutorial has been designed and tested using:
- Windows 10
- Visual Studio 2017 (toolset v141)
- mlpack-3.0.3
- OpenBLAS.0.2.14.1
- boost_1_66_0-msvc-14.1-64
- armadillo-8.500.1
- and x64 configuration

The directories and paths used in this tutorial are just for reference purposes.

@section build_windows_prereqs Pre-requisites

- Install CMake for Windows (win64-x64 version from https://cmake.org/download/)
and make sure you can use it from the Command Prompt (may need to add to the PATH)

- Download the latest mlpack release from here:
<a href="http://www.mlpack.org/download.html">mlpack</a>

@section build_windows_instructions Windows build instructions

- Unzip mlpack to "C:\mlpack\mlpack-3.0.3"
- Open Visual Studio and select: File > New > Project from Existing Code
 - Type of project: Visual C++
 - Project location: "C:\mlpack\mlpack-3.0.3"
 - Project name: mlpack
 - Finish
- We will use this Visual Studio project to get the OpenBLAS dependency in the next section

@section build_windows_dependencies Dependencies

<b> OpenBLAS Dependency </b>

- Open the NuGet packages manager (Tools > NuGet Package Manager > Manage NuGet Packages for Solution...)
- Click on the “Browse” tab and search for “openblas”
- Click on OpenBlas and check the mlpack project, then click Install
- Once it has finished installing, close Visual Studio

<b> Boost Dependency </b>

You can either get Boost via NuGet or you can download the prebuilt Windows binaries separately.
This tutorial follows the second approach for simplicity.

- Download the "Prebuilt Windows binaries" of the Boost library ("boost_1_66_0-msvc-14.1-64") from 
<a href="https://sourceforge.net/projects/boost/files/boost-binaries/">Sourceforge</a>

@note Make sure you download the MSVC version that matches your Visual Studio

- Install or unzip to "C:\boost\boost_1_66_0"

<b> Armadillo Dependency </b>

- Download "Armadillo" (armadillo-8.500.1.tar.xz) from <a href="http://arma.sourceforge.net/download.html">Sourceforge</a>
- Unzip to "C:\mlpack\armadillo-8.500.1"
- Create a "build" directory into "C:\mlpack\armadillo-8.500.1\"
- Open the Command Prompt and navigate to "C:\mlpack\armadillo-8.500.1\build"
- Run cmake: 

@code
cmake -G "Visual Studio 15 2017 Win64" -DBLAS_LIBRARY:FILEPATH="C:/mlpack/mlpack-3.0.3/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARY:FILEPATH="C:/mlpack/mlpack-3.0.3/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DCMAKE_PREFIX:FILEPATH="C:/mlpack/armadillo" -DBUILD_SHARED_LIBS=OFF ..
@endcode

@note If you are using different directory paths, a different configuration (e.g. Release)
or a different VS version, update the cmake command accordingly.

- Once it has successfully finished, open "C:\mlpack\armadillo-8.500.1\build\armadillo.sln"
- Build > Build Solution
- Once it has successfully finished, close Visual Studio

@section build_windows_mlpack Building mlpack

- Create a "build" directory into "C:\mlpack\mlpack-3.0.3\"
- Open the Command Prompt and navigate to "C:\mlpack\mlpack-3.0.3\build"
- Run cmake:

@code
cmake -G "Visual Studio 15 2017 Win64" -DBLAS_LIBRARY:FILEPATH="C:/mlpack/mlpack-3.0.3/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARY:FILEPATH="C:/mlpack/mlpack-3.0.3/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DARMADILLO_INCLUDE_DIR="C:/mlpack/armadillo-8.500.1/include" -DARMADILLO_LIBRARY:FILEPATH="C:/mlpack/armadillo-8.500.1/build/Debug/armadillo.lib" -DBOOST_INCLUDEDIR:PATH="C:/boost/boost_1_66_0/" -DBOOST_LIBRARYDIR:PATH="C:/boost/boost_1_66_0/lib64-msvc-14.1" -DDEBUG=OFF -DPROFILE=OFF ..
@endcode

- Once it has successfully finished, open "C:\mlpack\mlpack-3.0.3\build\mlpack.sln"
- Build > Build Solution (this may be by default in Debug mode)
- Once it has sucessfully finished, you will find the library files you need in: "C:\mlpack\mlpack-3.0.3\build\Debug" (or "C:\mlpack\mlpack-3.0.3\build\Release" if you changed to Release mode)

You are ready to create your first application, take a look at the @ref sample_ml_app "Sample C++ ML App"

*/
