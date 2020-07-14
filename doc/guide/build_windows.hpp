/**
 * @file build_windows.hpp
 * @author German Lancioni
 * @author Miguel Canteras
 * @author Shikhar Jaiswal
 * @author Ziyang Jiang

@page build_windows Building mlpack From Source on Windows

@section build_windows_intro Introduction

This tutorial will show you how to build mlpack for Windows from source, so you can
later create your own C++ applications. Before you try building mlpack, you may
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

@section build_windows_mlpack Building mlpack

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
    - If Boost is still not found, try adding the following variables and
      reconfigure:
      - Name: `Boost_INCLUDE_DIR`; type `PATH`; value `C:/boost/`
      - Name: `Boost_SERIALIZATION_LIBRARY_DEBUG`; type `FILEPATH`; value should be `C:/boost/lib64-msvc-14.2/boost_serialization-vc142-mt-gd-x64-1_71.lib`
      - Name: `Boost_SERIALIZATION_LIBRARY_RELEASE`; type `FILEPATH`; value should be `C:/boost/lib64-msvc-14.2/boost_program_options-vc142-mt-x64-1_71.lib`
      - Name: `Boost_UNIT_TEST_FRAMEWORK_LIBRARY_DEBUG`; type `FILEPATH`; value should be `C:/boost/lib64-msvc-14.2/boost_unit_test_framework-vc142-mt-gd-x64-1_71.lib`
      - Name: `Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE`; type `FILEPATH`; value should be `C:/boost/lib64-msvc-14.2/boost_unit_test_framework-vc142-mt-x64-1_71.lib`
    - Once CMake has configured successfully, hit "Generate" to create the `.sln` file.

@section build_windows_additional_information Additional Information

If you are facing issues during the build process of mlpack, you may take a look at other third-party tutorials for Windows, but they may be out of date:

 * <a href="https://github.com/mlpack/mlpack/wiki/WindowsBuild">Github wiki Windows Build page</a><br/>
 * <a href="http://keon.io/mlpack-on-windows">Keon's tutorial for mlpack 2.0.3</a><br/>
 * <a href="https://overdosedblog.wordpress.com/2016/08/15/once_again/">Kirizaki's tutorial for mlpack 2</a><br/>

*/
