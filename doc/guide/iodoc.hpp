/*! @page iodoc mlpack Input and Output

@section iointro Introduction

mlpack provides the following:

 - mlpack::Log, for debugging / informational / warning / fatal output
 - mlpack::CLI, for parsing command line options

Each of those classes are well-documented, and that documentation should be
consulted for further reference.

@section simplelog Simple Logging Example

mlpack has four logging levels:

 - Log::Debug
 - Log::Info
 - Log::Warn
 - Log::Fatal

Output to Log::Debug does not show (and has no performance penalty) when mlpack
is compiled without debugging symbols.  Output to Log::Info is only shown when
the program is run with the --verbose (or -v) flag.  Log::Warn is always shown,
and Log::Fatal will throw a std::runtime_error exception, when a newline is sent
to it only. If mlpack was compiled with debugging symbols, Log::Fatal will
always throw a std::runtime_error exception and print backtrace.

Here is a simple example, and its output:

@code
#include <mlpack/core.hpp>

using namespace mlpack;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  Log::Debug << "Compiled with debugging symbols." << std::endl;

  Log::Info << "Some test informational output." << std::endl;

  Log::Warn << "A warning!" << std::endl;

  Log::Fatal << "Program has crashed." << std::endl;

  Log::Warn << "Made it!" << std::endl;
}
@endcode

With debugging output--verbose, the following is shown:

@code
[DEBUG] Compiled with debugging symbols.
[INFO ] Some test informational output.
[WARN ] A warning!
[FATAL] [bt]: (1) /absolute/path/to/file/example.cpp:6: function()
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
@endcode

With debugging output, compilation flags -g -rdynamic and --verbose,
the following is shown:

@code
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

Without debugging symbols and without --verbose, the following is shown:

@code
$ ./main
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

@code
#include <mlpack/core.hpp>

// Document program.
PROGRAM_INFO("Principal Components Analysis", "This program performs principal "
    "components analysis on the given dataset.  It will transform the data "
    "onto its principal components, optionally performing dimensionality "
    "reduction by ignoring the principal components with the smallest "
    "eigenvalues.");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input dataset to perform PCA on.", "");
PARAM_STRING_REQ("output_file", "Output dataset to perform PCA on.", "");
PARAM_INT("new_dimensionality", "Desired dimensionality of output dataset.",
    "", 0);

using namespace mlpack;

int main(int argc, char** argv)
{
  // Parse commandline.
  CLI::ParseCommandLine(argc, argv);

  ...
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
  --output_file [string]        Output dataset to perform PCA on.

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
documentation.

*/
