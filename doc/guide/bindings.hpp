/*! @page bindings mlpack automatic bindings to other languages

@section bindings_overview Overview

mlpack has a system to automatically generate bindings to other languages, such
as Python and command-line programs, and it is extensible to other languages
with some amount of ease.  The maintenance burden of this system is low, and it
is designed in such a way that the bindings produced are always up to date
across languages and up to date with the mlpack library itself.

This document describes the full functioning of the system, and is a good place
to start for someone who wishes to understand the system so that they can
contribute a new binding language, or someone who wants to understand so they
can adapt the system for use in their own project, or someone who is simply
curious enough to see how the sausage is made.

@section bindings_intro Introduction

C++ is not the most popular language on the planet, and it (unfortunately) can
scare many away with its ultra-verbose error messages, confusing template rules,
and complex metaprogramming techniques.  Most practitioners of machine learning
tend to avoid writing native C++ and instead prefer other languages---probably
most notably Python.

In the case of Python, many projects will use tools like SWIG
(http://www.swig.org/) to automatically generate bindings, or they might
hand-write Cython.  The same types of strategies may be used for other
languages; hand-written MEX files may be used for MATLAB, hand-written RCpp
bindings might be used for R bindings, and so forth.

However, these approaches have a fundamental flaw: the hand-written bindings
must be maintained, and risk going out of date as the rest of the library
changes or new functionality is added.  This incurs a maintenance burden: each
major change to the library means that someone must update the bindings and test
that they are still working.  mlpack is not prepared to handle this maintenance
workload; therefore an alternate solution is needed.

At the time of the design of this system, mlpack shipped headers for a C++
library as well as many (~40) hand-written command-line programs that used the
mlpack::CLI object to manage command-line arguments.  These programs all had
similar structure, and could be logically split into three sections:

 - parse the input options supplied by the user
 - run the machine learning algorithm
 - prepare the output to return to the user

The user might interface with this command-line program like the following:

@code
$ mlpack_knn -r reference.csv -q query.csv -k 3 -d d.csv -n n.csv
@endcode

That is, they would pass a number of input options---some were numeric values
(like @c -k @c 3 ); some were filenames (like @c -r @c reference.csv ); and a
few other types also.  Therefore, the first stage of the program---parsing input
options---would be handled by reading the command line and loading any input
matrices.  Preparing the output, which usually consists of data matrices (i.e.
@c -d @c d.csv ) involves saving the matrix returned by the algorithm to the
user's desired file.

Ideally, any binding to any language would have this same structure, and the
actual "run the machine learning algorithm" code could be identical.  For
MATLAB, for instance, we would not need to read the file @c reference.csv but
instead the user would simply pass their data matrix as an argument.  So each
input and output parameter would need to be handled differently, but the
algorithm could be run identically across all bindings.

Therefore, design of an automatically-generated binding system would simply
involve generating the boilerplate code necessary to parse input options for a
given language, and to return output options to a user.

@section bindings_code Writing code that can be turned into a binding

This section details what a binding file might actually look like.  It is good
to have this API in mind when reading the following sections.

Each mlpack binding is typically contained in the @c src/mlpack/methods/ folder
corresponding to a given machine learning algorithm, with the suffix
@c _main.cpp ; so an example is @c src/mlpack/methods/pca/pca_main.cpp .

These files have roughly two parts:

 - definition of the input and output parameters with @c PARAM macros
 - implementation of @c mlpackMain(), which is the actual machine learning code

Here is a simple example file:

@code
// This is a stripped version of mean_shift_main.cpp.
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "mean_shift.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;
using namespace mlpack::kernel;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("Mean Shift Clustering", "This program performs mean shift "
    "clustering on the given dataset, storing the learned cluster assignments "
    "either as a column of labels in the file containing the input dataset or "
    "in a separate file.");

// Required option: the user must give us a matrix.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform clustering on.", "i");

// Output options: the user can save the output matrix of labels and/or the
// centroids.
PARAM_UCOL_OUT("output", "Matrix to write output labels to.", "o");
PARAM_MATRIX_OUT("centroid", "If specified, the centroids of each cluster will "
    "be written to the given matrix.", "C");

// Mean shift configuration options.
PARAM_INT_IN("max_iterations", "Maximum number of iterations before mean shift "
    "terminates.", "m", 1000);
PARAM_DOUBLE_IN("radius", "If the distance between two centroids is less than "
    "the given radius, one will be removed.  A radius of 0 or less means an "
    "estimate will be calculated and used for the radius.", "r", 0);

void mlpackMain()
{
  // Process the parameters that the user passed.
  const double radius = CLI::GetParam<double>("radius");
  const int maxIterations = CLI::GetParam<int>("max_iterations");

  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
        ")! Must be greater than or equal to 0." << endl;
  }

  // Warn, if the user did not specify that they wanted any output.
  if (!CLI::HasParam("output") && !CLI::HasParam("centroid"))
  {
    Log::Warn << "--output_file, --in_place, and --centroid_file are not set; "
        << "no results will be saved." << endl;
  }

  arma::mat dataset = std::move(CLI::GetParam<arma::mat>("input"));
  arma::mat centroids;
  arma::Col<size_t> assignments;

  // Prepare and run the actual algorithm.
  MeanShift<> meanShift(radius, maxIterations);

  Timer::Start("clustering");
  Log::Info << "Performing mean shift clustering..." << endl;
  meanShift.Cluster(dataset, assignments, centroids);
  Timer::Stop("clustering");

  Log::Info << "Found " << centroids.n_cols << " centroids." << endl;
  if (radius <= 0.0)
    Log::Info << "Estimated radius was " << meanShift.Radius() << ".\n";

  // Should we give the user the output matrix?
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::Col<size_t>>("output") = std::move(assignments);

  // Should we give the user the centroid matrix?
  if (CLI::HasParam("centroid"))
    CLI::GetParam<arma::mat>("centroid") = std::move(centroids);
}
@endcode

We can see that we have defined the basic program information in the
@c PROGRAM_INFO() macro.  This is, for instance, what is displayed to describe
the binding if the user passed the @c --help option for a command-line program.

Then, we define five parameters, three input and two output, that define the
data and options that the mean shift clustering will function on.  These
parameters are defined with the @c PARAM macros, of which there are many.  The
names of these macros specify the type, whether the parameter is required, and
whether the parameter is input or output.  Some examples:

 - @c PARAM_STRING_IN() -- a string-type input parameter
 - @c PARAM_MATRIX_OUT() -- a matrix-type output parameter
 - @c PARAM_DOUBLE_IN_REQ() -- a required double-type input parameter
 - @c PARAM_UMATRIX_IN() -- an unsigned matrix-type input parameter
 - @c PARAM_MODEL_IN() -- a serializable model-type input parameter

Note that each of these macros may have slightly different syntax.  See the
links above for further documentation.

@section bindings_general General structure of code

@section bindings_cli Command-line program bindings

@section bindings_python Python bindings

@section bindings_new Adding new binding types

*/
