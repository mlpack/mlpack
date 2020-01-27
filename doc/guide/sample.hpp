/*! @page sample Simple Sample mlpack Programs

@section sampleintro Introduction

On this page, several simple mlpack examples are contained, in increasing order
of complexity.  If you compile from the command-line, be sure that your compiler
is in C++11 mode.  With modern gcc and clang, this should already be the
default.

@note
The command-line programs like @c knn_main.cpp and @c
logistic_regression_main.cpp from the directory @c src/mlpack/methods/ cannot be
compiled easily by hand (the same is true for the individual tests in @c
src/mlpack/tests/); instead, those should be compiled with CMake, by running,
e.g., @c make @c mlpack_knn or @c make @c mlpack_test; see @ref build.  However,
any program that uses mlpack (and is not a part of the library itself) can be
compiled easily with g++ or clang from the command line.

@section compile_link_and_run Compile, Link and Run C++ Files 

Typically the following commands would be used to compile and run the C++ files.

@subsection compile_using_gcc Compile Using GCC

@code{.sh}
g++ -std=c++11 filename.cpp -o filename -lmlpack -larmadillo
./filename
@endcode

If @c pkg-config is installed:

@code{.sh}
g++ -std=c++11 filename.cpp -o filename \
`pkg-config --cflags --libs mlpack armadillo`
./filename
@endcode

@subsection compile_using_clang Compile Using Clang

@code{.sh}
clang++ -Wall filename.cpp -o filename -lmlpack -larmadillo
./filename
@endcode

If @c pkg-config is installed:

@code{.sh}
clang++ -Wall filename.cpp -o filename \
`pkg-config --cflags --libs mlpack armadillo`
./filename
@endcode

@note
Flags like @c -lboost_serialization might be required depending upon the 
code. If you are using LAPACK and BLAS instead of Armadillo the @c -llapack
and @c -lblas would be required instead of @c -larmadillo

@subsection compile_using_cmake Compile Using Cmake

In /src
  - filename.cpp
  - CMakeLists.txt
  - /cmake/FindArmadillo.cmake
  - /cmake/FindMLPACK.cmake

@note
Create folders @c /src and @c /src/cmake . 
Place the files @c FindArmadillo.cmake and @c FindMLPACK.cmake 
in @c /src/cmake .
. The required files could be found at 
 - <a href="https://github.com/mlpack/mlpack/blob/master/CMake/FindArmadillo.cmake"> 
      FindArmadillo.cmake</a> and
 - <a href="https://github.com/mlpack/models/blob/master/CMake/FindMLPACK.cmake"> 
      FindMLPACK.cmake</a>
      
@code
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# set the project name
project(Project_Name VERSION 1.0)

# specify the search path for CMake modules to be loaded by find_package()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Armadillo REQUIRED)
find_package(MLPACK REQUIRED)

#List of preprocessor include file search directories
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${MLPACK_INCLUDE_DIRS})

add_executable(Project_Name filename.cpp)

#Specify libraries to use when linking a target Project_Name
target_link_libraries(Project_Name ${ARMADILLO_LIBRARIES})
target_link_libraries(Project_Name ${MLPACK_LIBRARIES})
@endcode

In target directory for executable, run the command:
@code{.sh}
cmake path/to/src 
cmake --build .
./Project_Name
@endcode

@section covariance Covariance Computation

A simple program to compute the covariance of a data matrix ("data.csv"),
assuming that the data is already centered, and save it to file.

@code
// Includes all relevant components of mlpack.
#include <mlpack/core.hpp>

// Convenience.
using namespace mlpack;

int main()
{
  // First, load the data.
  arma::mat data;
  // Use data::Load() which transposes the matrix.
  data::Load("data.csv", data, true);

  // Now compute the covariance.  We assume that the data is already centered.
  // Remember, because the matrix is column-major, the covariance operation is
  // transposed.
  arma::mat cov = data * trans(data) / data.n_cols;

  // Save the output.
  data::Save("cov.csv", cov, true);
}
@endcode

@section nn Nearest Neighbor

This simple program uses the mlpack::neighbor::NeighborSearch object to find the
nearest neighbor of each point in a dataset using the L1 metric, and then print
the index of the neighbor and the distance of it to stdout.

@code
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance

int main()
{
  // Load the data from data.csv (hard-coded).  Use CLI for simple command-line
  // parameter handling.
  arma::mat data;
  data::Load("data.csv", data, true);

  // Use templates to specify that we want a NeighborSearch object which uses
  // the Manhattan distance.
  NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);

  // Create the object we will store the nearest neighbors in.
  arma::Mat<size_t> neighbors;
  arma::mat distances; // We need to store the distance too.

  // Compute the neighbors.
  nn.Search(1, neighbors, distances);

  // Write each neighbor and distance using Log.
  for (size_t i = 0; i < neighbors.n_elem; ++i)
  {
    std::cout << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
  }
}
@endcode

@section other Other examples

For more complex examples, it is useful to refer to the main executables, found
in @c src/mlpack/methods/.  A few are listed below.

 - methods/neighbor_search/knn_main.cpp
 - methods/neighbor_search/kfn_main.cpp
 - methods/emst/emst_main.cpp
 - methods/radical/radical_main.cpp
 - methods/nca/nca_main.cpp
 - methods/naive_bayes/nbc_main.cpp
 - methods/pca/pca_main.cpp
 - methods/lars/lars_main.cpp
 - methods/linear_regression/linear_regression_main.cpp
 - methods/gmm/gmm_main.cpp
 - methods/kmeans/kmeans_main.cpp

*/
