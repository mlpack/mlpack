/*! @page iodoc MLPACK Input and Output

@section iointro Introduction

MLPACK provides the following:

 - mlpack::Log, for debugging / informational / warning / fatal output
 - mlpack::CLI, for parsing command line options

Each of those classes are well-documented, and that documentation should be
consulted for further reference.

@section simplelog Simple Logging Example

MLPACK has four logging levels:

 - Log::Debug
 - Log::Info
 - Log::Warn
 - Log::Fatal

Output to Log::Debug does not show (and has no performance penalty) when MLPACK
is compiled without debugging symbols.  Output to Log::Info is only shown when
the program is run with the --verbose (or -v) flag.  Log::Warn is always shown,
and Log::Fatal will halt the program, when a newline is sent to it.

Here is a simple example, and its output:

@code
#include <mlpack/core.hpp>

using namespace mlpack;

int main()
{
  Log::Debug << "Compiled with debugging symbols." << std::endl;

  Log::Info << "Some test informational output." << std::endl;

  Log::Warn << "A warning!" << std::endl;

  Log::Fatal << "Program has crashed." << std::endl;

  Log::Warn << "Made it!" << std::endl;
}
@endcode

With debugging output and --verbose, the following is shown:

@code
$ ./main --verbose
[DEBUG] Compiled with debugging symbols.
[INFO ] Some test informational output.
[WARN ] A warning!
[FATAL] Program has crashed.
@endcode

The last warning is not reached, because Log::Fatal terminates the program.

Without debugging symbols and without --verbose, the following is shown:

@code
$ ./main
[WARN ] A warning!
[FATAL] Program has crashed.
@endcode

These four outputs can be very useful for both providing informational output
and debugging output for your MLPACK program.

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
$ pca --help
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
