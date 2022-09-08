# Writing an mlpack binding

This tutorial gives some simple examples of how to write an mlpack binding that
can be compiled for multiple languages.  These bindings make up the core of how
most users will interact with mlpack.

mlpack provides the following:

 - `mlpack::Log`, for debugging / informational / warning / fatal output
 - a `util::Params` object, for parsing command line options or other option
 - a `util::Timers` object, for collecting and displaying timing information

Each of those classes are well-documented, and that documentation in the source
code should be consulted for further reference.

First, we'll discuss the logging infrastructure, which is useful for giving
output that users can see.

## Simple logging example

mlpack has four logging levels:

 - `Log::Debug`
 - `Log::Info`
 - `Log::Warn`
 - `Log::Fatal`

Output to `Log::Debug` does not show (and has no performance penalty) when
mlpack is compiled without debugging symbols.  Output to `Log::Info` is only
shown when the program is run with the `verbose` option (for a command-line
binding, this is `--verbose` or `-v`).  `Log::Warn` is always shown, and
`Log::Fatal` will throw a `std::runtime_error` exception, after a newline is
sent to it. If mlpack was compiled with debugging symbols, `Log::Fatal` will
also print a backtrace, if the necessary libraries are available.

Here is a simple example binding, and its output.  Note that instead of
`int main()`, we use `void BINDING_FUNCTION()`.  This is because the
[automatic binding generator](bindings.md) will set up the environment and
once that is done, it will call `BINDING_FUNCTION()`.

```c++
#include <mlpack/core.hpp>
#include <mlpack/core/util/io.hpp>
// This definition below means we will only compile for the command line.
#define BINDING_TYPE BINDING_TYPE_CLI
#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  Log::Debug << "Compiled with debugging symbols." << std::endl;

  Log::Info << "Some test informational output." << std::endl;

  Log::Warn << "A warning!" << std::endl;

  Log::Fatal << "Program has crashed." << std::endl;

  Log::Warn << "Made it!" << std::endl;
}
```

Assuming mlpack is installed on the system and the code above is saved in
`test.cpp`, this program can be compiled with the following command:

```sh
$ g++ -o test test.cpp -DDEBUG -g -rdynamic -lmlpack
```

Since we compiled with `-DDEBUG`, if we run the program as below, the following
output is shown:

```sh
$ ./test --verbose
[DEBUG] Compiled with debugging symbols.
[INFO ] Some test informational output.
[WARN ] A warning!
[FATAL] [bt]: (1) /absolute/path/to/file/example.cpp:6: function()
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
```

The flags `-g` and `-rdynamic` are only necessary for providing a backtrace.
If those flags are not given during compilation, the following output would be
shown:

```sh
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
```

The last warning is not reached, because `Log::Fatal` terminates the program.

Without debugging symbols (i.e. without `-g` and `-DDEBUG`) and without
`--verbose`, the following is shown:

```sh
$ ./test
[WARN ] A warning!
[FATAL] Program has crashed.
terminate called after throwing an instance of 'std::runtime_error'
  what():  fatal error; see Log::Fatal output
Aborted
```

These four outputs can be very useful for both providing informational output
and debugging output for your mlpack program.

## Simple parameter example

Through the `mlpack::util::Params` object, parameters can be easily added to a
binding with the `BINDING_NAME`, `BINDING_SHORT_DESC`, `BINDING_LONG_DESC`,
`BINDING_EXAMPLE`, `BINDING_SEE_ALSO`, `PARAM_INT`, `PARAM_DOUBLE`,
`PARAM_STRING`, and `PARAM_FLAG` macros.

Here is a sample use of those macros, extracted from `methods/pca/pca_main.cpp`.
(Some details have been omitted from the snippet below.)

```c++
#include <mlpack/core.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

// Program Name.
BINDING_NAME("Principal Components Analysis");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of several strategies for principal components analysis "
    "(PCA), a common preprocessing step.  Given a dataset and a desired new "
    "dimensionality, this can reduce the dimensionality of the data using the "
    "linear transformation determined by PCA.");

// Long description.
BINDING_LONG_DESC(
    "This program performs principal components analysis on the given dataset "
    "using the exact, randomized, randomized block Krylov, or QUIC SVD method. "
    "It will transform the data onto its principal components, optionally "
    "performing dimensionality reduction by ignoring the principal components "
    "with the smallest eigenvalues.");

// See also...
BINDING_SEE_ALSO("Principal component analysis on Wikipedia",
    "https://en.wikipedia.org/wiki/Principal_component_analysis");
BINDING_SEE_ALSO("PCA C++ class documentation",
    "@src/mlpack/methods/pca/pca.hpp");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform PCA on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("new_dimensionality", "Desired dimensionality of output dataset.",
    "d", 0);

using namespace mlpack;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Load input dataset.
  arma::mat& dataset = params.Get<arma::mat>("input");

  size_t newDimension = params.Get<int>("new_dimensionality");

  ...

  // Now save the results.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(dataset);
}
```

Documentation is automatically generated using those macros, and if compiled to
a command-line program, when that program is run with `--help` the following is
displayed:

```
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
```

The `mlpack::IO` source code can be consulted for further and complete
documentation.  Also useful is to look at other example bindings, found in
`src/mlpack/methods/`.
