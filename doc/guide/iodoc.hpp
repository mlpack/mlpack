/*! @page iodoc Writing an mlpack binding

@section iointro Introduction

This tutorial gives some simple examples of how to write an mlpack binding that
can be compiled for multiple languages.  These bindings make up the core of how
most users will interact with mlpack.

mlpack provides the following:

 - mlpack::Log, for debugging / informational / warning / fatal output
 - mlpack::CLI, for parsing command line options or other option

Each of those classes are well-documented, and that documentation should be
consulted for further reference.

First, we'll discuss the logging infrastructure, which is useful for giving
output that users can see.

@section simplelog Simple Logging Example

mlpack has four logging levels:

 - Log::Debug
 - Log::Info
 - Log::Warn
 - Log::Fatal

Output to Log::Debug does not show (and has no performance penalty) when mlpack
is compiled without debugging symbols.  Output to Log::Info is only shown when
the program is run with the \c --verbose (or \c -v) flag.  Log::Warn is always
shown, and Log::Fatal will throw a std::runtime_error exception, after a newline
is sent to it. If mlpack was compiled with debugging symbols, Log::Fatal will
also print a backtrace, if the necessary libraries are available.

Here is a simple example binding, and its output.  Note that instead of
\c int \c main(), we use \c static \c void \c mlpackMain().  This is because the
automatic binding generator (see \ref bindings) will set up the environment and
once that is done, it will call \c mlpackMain().

@code
#include <mlpack/core.hpp>
#include <mlpack/core/util/cli.hpp>
// This definition below means we will only compile for the CLI.
#define BINDING_TYPE BINDING_TYPE_CLI
#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;

static void mlpackMain()
{
  Log::Debug << "Compiled with debugging symbols." << std::endl;

  Log::Info << "Some test informational output." << std::endl;

  Log::Warn << "A warning!" << std::endl;

  Log::Fatal << "Program has crashed." << std::endl;

  Log::Warn << "Made it!" << std::endl;
}
@endcode

Assuming mlpack is installed on the system and the code above is saved in
\c test.cpp, this program can be compiled with the following command:

@code
$ g++ -o test test.cpp -DDEBUG -g -rdynamic -lmlpack
@endcode

Since we compiled with \c -DDEBUG, if we run the program as below, the following
output is shown:

@code
$ ./test --verbose
[DEBUG] Compiled with debugging symbols.
[INFO ] Some test informational output.
[WARN ] A warning!
[FATAL] [bt]: (1) /absolute/path/to/file/example.cpp:6: function()
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
@endcode

The flags \c -g and \c -rdynamic are only necessary for providing a backtrace.
If those flags are not given during compilation, the following output would be
shown:

@code
$ ./test --verbose
[DEBUG] Compiled with debugging symbols.
[INFO ] Some test informational output.
[WARN ] A warning!
[FATAL] Cannot give backtrace because program was compiled without: -g -rdynamic
[FATAL] For a backtrace, recompile with: -g -rdynamic.
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
@endcode

The last warning is not reached, because Log::Fatal terminates the program.

Without debugging symbols (i.e. without \c -g and \c -DDEBUG) and without
--verbose, the following is shown:

@code
$ ./test
[WARN ] A warning!
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
@endcode

These four outputs can be very useful for both providing informational output
and debugging output for your mlpack program.

@section simplecli Simple CLI Example

Through the mlpack::CLI object, command-line parameters can be easily added
with the PROGRAM_INFO, PARAM_INT, PARAM_DOUBLE, PARAM_STRING, and PARAM_FLAG
macros.

Here is a sample use of those macros, extracted from methods/pca/pca_main.cpp.
(Some details have been omitted from the snippet below.)

@code
#include <mlpack/core.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

// Document program.
PROGRAM_INFO("Principal Components Analysis", "This program performs principal "
    "components analysis on the given dataset.  It will transform the data "
    "onto its principal components, optionally performing dimensionality "
    "reduction by ignoring the principal components with the smallest "
    "eigenvalues.");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform PCA on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("new_dimensionality", "Desired dimensionality of output dataset.",
    "d", 0);

using namespace mlpack;

static void mlpackMain()
{
  // Load input dataset.
  arma::mat& dataset = CLI::GetParam<arma::mat>("input");

  size_t newDimension = CLI::GetParam<int>("new_dimensionality");

  ...

  // Now save the results.
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(dataset);
}
@endcode

Documentation is automatically generated using those macros, and when the
program is run with --help the following is displayed:

@code
$ mlpack_pca --help
Principal Components Analysis

  This program performs principal components analysis on the given dataset.  It
  will transform the data onto its principal components, optionally performing
  dimensionality reduction by ignoring the principal components with the
  smallest eigenvalues.

Required options:

  --input_file [string]         Input dataset to perform PCA on.
  --output_file [string]        Matrix to save modified dataset to.

Options:

  --help (-h)                   Default help info.
  --info [string]               Get help on a specific module or option.
                                Default value ''.
  --new_dimensionality [int]    Desired dimensionality of output dataset.
                                Default value 0.
  --verbose (-v)                Display informational messages and the full list
                                of parameters and timers at the end of
                                execution.
@endcode

The mlpack::CLI documentation can be consulted for further and complete
documentation.  Also useful is to look at other example bindings, found in
\c src/mlpack/methods/.

*/
