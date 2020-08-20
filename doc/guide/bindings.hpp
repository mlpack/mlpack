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

The document is split into several sections:

 - @ref bindings_intro
 - @ref bindings_code
 - @ref bindings_general
    - @ref bindings_general_program_doc
    - @ref bindings_general_define_params
    - @ref bindings_general_functions
    - @ref bindings_general_more
 - @ref bindings_structure
 - @ref bindings_cli
    - @ref bindings_cli_mlpack_main
    - @ref bindings_cli_matrix
    - @ref bindings_cli_parsing
 - @ref bindings_python
    - @ref bindings_python_matrix
    - @ref bindings_python_model
    - @ref bindings_python_setup_py
    - @ref bindings_python_build_pyx
    - @ref bindings_python_testing
 - @ref bindings_new

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
mlpack::IO object to manage command-line arguments.  These programs all had
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

// Define the help text for the program.  The PRINT_PARAM_STRING() and
// PRINT_DATASET() macros are used to print the name of the parameter as seen in
// the binding type that is being used, and the PRINT_CALL() macro generates a
// sample invocation of the program in the language of the binding type that is
// being used.  Note that the macros must have + on either side of them.  We
// provide some extra references with the "SEE_ALSO()" macro, which is used to
// generate documentation for the website.

// Program Name.
BINDING_NAME("Mean Shift Clustering");

// Short description.
BINDING_SHORT_DESC(
    "A fast implementation of mean-shift clustering using dual-tree range "
    "search.  Given a dataset, this uses the mean shift algorithm to produce "
    "and return a clustering of the data.");

// Long description.
BINDING_LONG_DESC(
    "This program performs mean shift clustering on the given dataset, storing "
    "the learned cluster assignments either as a column of labels in the input "
    "dataset or separately."
    "\n\n"
    "The input dataset should be specified with the " +
    PRINT_PARAM_STRING("input") + " parameter, and the radius used for search"
    " can be specified with the " + PRINT_PARAM_STRING("radius") + " "
    "parameter.  The maximum number of iterations before algorithm termination "
    "is controlled with the " + PRINT_PARAM_STRING("max_iterations") + " "
    "parameter."
    "\n\n"
    "The output labels may be saved with the " + PRINT_PARAM_STRING("output") +
    " output parameter and the centroids of each cluster may be saved with the"
    " " + PRINT_PARAM_STRING("centroid") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to run mean shift clustering on the dataset " +
    PRINT_DATASET("data") + " and store the centroids to " +
    PRINT_DATASET("centroids") + ", the following command may be used: "
    "\n\n" +
    PRINT_CALL("mean_shift", "input", "data", "centroid", "centroids"));

// See also...
BINDING_SEE_ALSO("@kmeans", "#kmeans");
BINDING_SEE_ALSO("@dbscan", "#dbscan");
BINDING_SEE_ALSO("Mean shift on Wikipedia",
        "https://en.wikipedia.org/wiki/Mean_shift");
BINDING_SEE_ALSO("Mean Shift, Mode Seeking, and Clustering (pdf)",
        "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.510.1222"
        "&rep=rep1&type=pdf");
BINDING_SEE_ALSO("mlpack::mean_shift::MeanShift C++ class documentation",
        "@doxygen/classmlpack_1_1meanshift_1_1MeanShift.html");

// Define parameters for the executable.

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
  const double radius = IO::GetParam<double>("radius");
  const int maxIterations = IO::GetParam<int>("max_iterations");

  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
        ")! Must be greater than or equal to 0." << endl;
  }

  // Warn, if the user did not specify that they wanted any output.
  if (!IO::HasParam("output") && !IO::HasParam("centroid"))
  {
    Log::Warn << "--output_file, --in_place, and --centroid_file are not set; "
        << "no results will be saved." << endl;
  }

  arma::mat dataset = std::move(IO::GetParam<arma::mat>("input"));
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
  if (IO::HasParam("output"))
    IO::GetParam<arma::Col<size_t>>("output") = std::move(assignments);

  // Should we give the user the centroid matrix?
  if (IO::HasParam("centroid"))
    IO::GetParam<arma::mat>("centroid") = std::move(centroids);
}
@endcode

We can see that we have defined the basic program information in the
@c BINDING_NAME(), @c BINDING_SHORT_DESC(), @c BINDING_LONG_DESC(),
@c BINDING_EXAMPLE() and @c BINDING_SEE_ALSO() macros.  This is, for instance,
what is displayed to describe the binding if the user passed the
<tt>\--help</tt> option for a command-line program.

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

In order to write a new binding, then, you simply must write @c BINDING_NAME(),
@c BINDING_SHORT_DESC(), @c BINDING_LONG_DESC(), @c BINDING_EXAMPLE() and
@c BINDING_SEE_ALSO() definitions of the program with some docuentation, define
the input and output parameters as @c PARAM macros, and then write an
@c mlpackMain() function that actually performs the functionality of the binding.
Inside of @c mlpackMain():

 - All input parameters are accessible through @c IO::GetParam<type>("name").
 - All output parameters should be set by the end of the function with the
      @c IO::GetParam<type>("name") method.

Then, assuming that your program is saved in the file @c program_name_main.cpp,
generating bindings for other languages is a simple addition to the
@c CMakeLists.txt file:

@code
add_cli_executable(program_name)
add_python_binding(program_name)
add_markdown_docs(program_name "cli;python" "category")
@endcode

In this example, @c add_markdown_docs() will generate documentation that is
typically used to build the website.  The "category" parameter should be one of
the categories in @c src/mlpack/bindings/markdown/MarkdownCategories.cmake.

@section bindings_general How to write mlpack bindings

This section describes the general structure of the @c IO code and how one
might write a new binding for mlpack.  After reading this section it should be
relatively clear how one could use the @c IO functionality along with CMake to
add a binding for a new mlpack machine learning method.  If it is not clear,
then the examples in the following sections should clarify.

@subsection bindings_general_program_doc Documenting a program with
@c BINDING_NAME(), @c BINDING_SHORT_DESC(), @c BINDING_LONG_DESC(),
@c BINDING_EXAMPLE() and @c BINDING_SEE_ALSO().

Any mlpack program should be documented with the @c BINDING_NAME(),
@c BINDING_SHORT_DESC(), @c BINDING_LONG_DESC() , @c BINDING_EXAMPLE() and
@c BINDING_SEE_ALSO() macros, which is available from the
@c <mlpack/core/util/mlpack_main.hpp> header.  The macros
are of the form

@code
BINDING_NAME("program name");
BINDING_SHORT_DESC("This is a short, two-sentence description of what the program does.");
BINDING_LONG_DESC("This is a long description of what the program does."
    " It might be many lines long and have lots of details about different options.");
BINDING_EXAMPLE("This contains one example for this particular binding.\n" +
    PROGRAM_CALL(...));
BINDING_EXAMPLE("This contains another example for this particular binding.\n" +
    PROGRAM_CALL(...));
// There could be many of these "see alsos".
BINDING_SEE_ALSO("https://en.wikipedia.org/wiki/Machine_learning");
@endcode

The short documentation should be two sentences indicating what the program
implements and does, and a quick overview of how it can be used and what it
should be used for.  When writing new short documentation, it is a good idea to
take a look at the existing documentation to get an idea of the general format.

For the "see also" section, you can specify as many @c SEE_ALSO() calls as you
see fit.  These are links used at the "see also" section of the website
documentation for each binding, and it's very important that relevant links are
provided (also to other bindings).  See the @c SEE_ALSO() documentation for more
details.

Although it is possible to provide very short documentation, it is certainly
better to provide a long description including

 - what the program does
 - a basic overview of what input and output parameters the program has
 - at least one example invocation

Examples are very important, and are probably what most users are going to
immediately search for, instead of taking a long time to read and carefully
consider all of the written documentation.

However, it is difficult to write language-agnostic documentation.  For
instance, in a command-line program, an output parameter '\--output_file' would
be specified on the command line as an input parameter, but in Python, the
output parameter 'output' would actually simply be returned from the call to the
Python function.  Therefore, we must be careful how our documentation refers to
input and output parameters.  The following general guidelines can help:

 - Always refer to output parameters as "output parameters", which is a fairly
   close term that can be interpreted to mean both "return values" for languages
   like Python and MATLAB and also "arguments given on the command line" for
   command line programs.

 - Use the provided @c PRINT_PARAM_STRING() macro to print the names of
   parameters.  For instance, <tt>PRINT_PARAM_STRING("shuffle")</tt> will print
   @c '\--shuffle' for a command line program and @c 'shuffle' for a Python
   binding.  The @c PRINT_PARAM_STRING() macro also takes into account the type
   of the parameter.

 - Use the provided @c PRINT_DATASET() and @c PRINT_MODEL() macro to introduce
   example datasets or models, which can be useful when introducing an example
   usage of the program.  So you could write @c '"to @c run @c with @c a
   @c dataset @c " @c + @c PRINT_DATASET("data") @c + @c "..."'.

 - Use the provided @c PRINT_CALL() macro to print example invocations of the
   program.  The first argument is the name of the program, and then the
   following arguments should be the name of a parameter followed by the value
   of that parameter.

 - Never mention files in the documentation---files are only relevant to
   command-line programs.  Similarly, avoid mentioning anything
   language-specific.

 - Remember that some languages give output through return values and some give
   output using other input parameters.  So the right verbiage to use is, e.g.,
   <tt>'the results may be saved using the PRINT_PARAM_STRING("output")
   parameter'</tt>, and @b not <tt>'the results are returned through the
   PRINT_PARAM_STRING("output") parameter'</tt>.

Each of these macros (@c PRINT_PARAM_STRING(), @c PRINT_DATASET(),
@c PRINT_MODEL(), and @c PRINT_CALL() ) provides different output depending on
the language.  Below are some example of documentation strings and their outputs
for different languages.  Note that the output might not be *exactly* as written
or formatted here, but the general gist should be the same.

@code
Input C++ (snippet):

  "The parameter " + PRINT_PARAM_STRING("shuffle") + ", if set, will shuffle "
  "the data before learning."

Command-line program output (snippet):

  The parameter '--shuffle', if set, will shuffle the data before learning.

Python binding output (snippet):

  The parameter 'shuffle', if set, will shuffle the data before learning.

Julia binding output (snippet):

  The parameter `shuffle`, if set, will shuffle the data before learning.

Go binding output (snippet):

  The parameter "Shuffle", if set, will shuffle the data before learning.
@endcode

@code
Input C++ (snippet):

  "The output matrix can be saved with the " + PRINT_PARAM_STRING("output") +
  " output parameter."

Command-line program output (snippet):

  The output matrix can be saved with the '--output_file' output parameter.

Python binding output (snippet):

  The output matrix can be saved with the 'output' output parameter.

Julia binding output (snippet):

  The output matrix can be saved with the `output` output parameter.

Go binding output (snippet):

  The output matrix can be saved with the "output" output parameter.
@endcode

@code
Input C++ (snippet):

  "For example, to train a model on the dataset " + PRINT_DATASET("x") + " and "
  "save the output model to " + PRINT_MODEL("model") + ", the following command"
  " can be used:"
  "\n\n" +
  PRINT_CALL("program", "input", "x", "output_model", "model")

Command-line program output (snippet):

  For example, to train a model on the dataset 'x.csv' and save the output model
  to 'model.bin', the following command can be used:

  $ program --input_file x.csv --output_model_file model.bin

Python binding output (snippet):

  For example, to train a model on the dataset 'x' and save the output model to
  'model', the following command can be used:

  >>> output = program(input=x)
  >>> model = output['output_model']

Julia binding output (snippet):

  For example, to train a model on the dataset `x` and save the output model to
  `model`, the following command can be used:

  julia> model = program(input=x)

Go binding output (snippet):

  For example, to train a model on the dataset "x" and save the output model to
  "model", the following command can be used:

    // Initialize optional parameters for Program().
    param := mlpack.ProgramOptions()
    param.Input = x

    model := mlpack.Program(param)
@endcode

@code
Input C++ (full program, 'random_numbers_main.cpp'):

  // Program Name.
  BINDING_NAME("Random Numbers");

  // Short description.
  BINDING_SHORT_DESC("An implementation of Random Numbers");

  // Long description.
  BINDING_LONG_DESC(
      "This program generates random numbers with a "
      "variety of nonsensical techniques and example parameters.  The input "
      "dataset, which will be ignored, can be specified with the " +
      PRINT_PARAM_STRING("input") + " parameter.  If you would like to subtract"
      " values from each number, specify the " +
      PRINT_PARAM_STRING("subtract") + " parameter.  The number of random "
      "numbers to generate is specified with the " +
      PRINT_PARAM_STRING("num_values") + " parameter."
      "\n\n"
      "The output random numbers can be saved with the " +
      PRINT_PARAM_STRING("output") + " output parameter.  In addition, a "
      "randomly generated linear regression model can be saved with the " +
      PRINT_PARAM_STRING("output_model") + " output parameter.");

  // Example.
  BINDING_EXAMPLE(
      "For example, to generate 100 random numbers with 3 subtracted from them "
      "and save the output to " + PRINT_DATASET("rand") + " and the random "
      "model to " + PRINT_MODEL("rand_lr") + ", use the following "
      "command:"
      "\n\n" +
      PRINT_CALL("random_numbers", "num_values", 100, "subtract", 3, "output",
          "rand", "output_model", "rand_lr"));

Command line output:

    Random Numbers

    This program generates random numbers with a variety of nonsensical
    techniques and example parameters.  The input dataset, which will be
    ignored, can be specified with the '--input_file' parameter.  If you would
    like to subtract values from each number, specify the '--subtract'
    parameter.  The number of random numbers to generate is specified with the
    '--num_values' parameter.

    The output random numbers can be saved with the '--output_file' output
    parameter.  In addition, a randomly generated linear regression model can be
    saved with the '--output_model_file' output parameter.

    For example, to generate 100 random numbers with 3 subtracted from them and
    save the output to 'rand.csv' and the random model to 'rand_lr.bin', use the
    following command:

    $ random_numbers --num_values 100 --subtract 3 --output_file rand.csv
      --output_model_file rand_lr.bin

Python binding output:

    Random Numbers

    This program generates random numbers with a variety of nonsensical
    techniques and example parameters.  The input dataset, which will be
    ignored, can be specified with the 'input' parameter.  If you would like to
    subtract values from each number, specify the 'subtract' parameter.  The
    number of random numbers to generate is specified with the 'num_values'
    parameter.

    The output random numbers can be saved with the 'output' output parameter.
    In addition, a randomly generated linear regression model can be saved with
    the 'output_model' output parameter.

    For example, to generate 100 random numbers with 3 subtracted from them and
    save the output to 'rand' and the random model to 'rand_lr', use the
    following command:

    >>> output = random_numbers(num_values=100, subtract=3)
    >>> rand = output['output']
    >>> rand_lr = output['output_model']

Julia binding output:

    Random Numbers

    This program generates random numbers with a variety of nonsensical
    techniques and example parameters.  The input dataset, which will be
    ignored, can be specified with the `input` parameter.  If you would like to
    subtract values from each number, specify the `subtract` parameter.  The
    number of random numbers to generate is specified with the `num_values`
    parameter.

    The output random numbers can be saved with the `output` output parameter.
    In addition, a randomly generated linear regression model can be saved with
    the `output_model` output parameter.

    For example, to generate 100 random numbers with 3 subtracted from them and
    save the output to `rand` and the random model to `rand_lr`, use the
    following command:

    ```julia
    julia> rand, rand_lr = random_numbers(num_values=100, subtract=3)
    ```

Go binding output:

    Random Numbers

    This program generates random numbers with a variety of nonsensical
    techniques and example parameters.  The input dataset, which will be
    ignored, can be specified with the "Input" parameter.  If you would like to
    subtract values from each number, specify the "Subtract" parameter.  The
    number of random numbers to generate is specified with the "NumValues"
    parameter.

    The output random numbers can be saved with the "output" output parameter.
    In addition, a randomly generated linear regression model can be saved with
    the "outputModel" output parameter.

    For example, to generate 100 random numbers with 3 subtracted from them and
    save the output to "rand" and the random model to "randLr", use the
    following command:

    // Initialize optional parameters for RandomNumbers().
    param := mlpack.RandomNumbersOptions()
    param.NumValues = 100
    param.Subtract=3

    rand, randLr := mlpack.RandomNumbers(param)
@endcode

@subsection bindings_general_define_params Defining parameters for a program

There exist several macros that can be used after a @c BINDING_LONG_DESC() and
@c BINDING_EXAMPLE() definition to define the parameters that can be specified
for a given mlpack program. These macros all have the same general definition:
the name of the macro specifies the type of the parameter, whether or not the
parameter is required, and whether the parameter is an input or output parameter.
Then as arguments to the macros, the name, description, and sometimes the
single-character alias and the default value of the parameter.

To give a flavor of how these definitions look, the definition

@code
PARAM_STRING_IN("algorithm", "The algorithm to use: 'svd' or 'blah'.", "a");
@endcode

will define a string input parameter @c algorithm (referenced as
@c '\--algorithm' from the command-line or @c 'algorithm' from Python) with the
description <tt>The algorithm to use: 'svd' or 'blah'.</tt>  The
single-character alias @c '-a' can be used from a command-line program (but
means nothing in Python).

There are numerous different macros that can be used:

 - @c PARAM_FLAG() - boolean flag parameter
 - @c PARAM_INT_IN() - integer input parameter
 - @c PARAM_INT_OUT() - integer output parameter
 - @c PARAM_DOUBLE_IN() - double input parameter
 - @c PARAM_DOUBLE_OUT() - double output parameter
 - @c PARAM_STRING_IN() - string input parameter
 - @c PARAM_STRING_OUT() - string output parameter
 - @c PARAM_MATRIX_IN() - double-valued matrix (<tt>arma::mat</tt>) input
       parameter
 - @c PARAM_MATRIX_OUT() - double-valued matrix (<tt>arma::mat</tt>) output
       parameter
 - @c PARAM_UMATRIX_IN() - size_t-valued matrix (<tt>arma::Mat<size_t></tt>)
       input parameter
 - @c PARAM_UMATRIX_OUT() - size_t-valued matrix (<tt>arma::Mat<size_t></tt>)
       output parameter
 - @c PARAM_TMATRIX_IN() - transposed double-valued matrix (<tt>arma::mat</tt>)
       input parameter
 - @c PARAM_TMATRIX_OUT() - transposed double-valued matrix (<tt>arma::mat</tt>)
       output parameter
 - @c PARAM_MATRIX_AND_INFO_IN() - matrix with categoricals input parameter
       (<tt>std::tuple<data::DatasetInfo, arma::mat</tt>)
 - @c PARAM_COL_IN() - double-valued column vector (<tt>arma::vec</tt>) input
       parameter
 - @c PARAM_COL_OUT() - double-valued column vector (<tt>arma::vec</tt>) output
       parameter
 - @c PARAM_UCOL_IN() - size_t-valued column vector (<tt>arma::Col<size_t></tt>)
       input parameter
 - @c PARAM_UCOL_OUT() - size_t-valued column vector
       (<tt>arma::Col<size_t></tt>) output parameter
 - @c PARAM_ROW_IN() - double-valued row vector (<tt>arma::rowvec</tt>) input
       parameter
 - @c PARAM_ROW_OUT() - double-valued row vector (<tt>arma::rowvec</tt>) output
       parameter
 - @c PARAM_VECTOR_IN() - <tt>std::vector</tt> input parameter
 - @c PARAM_VECTOR_OUT() - <tt>std::vector</tt> output parameter
 - @c PARAM_MODEL_IN() - serializable model input parameter
 - @c PARAM_MODEL_OUT() - serializable model output parameter

And for input parameters, the parameter may also be required:

 - @c PARAM_INT_IN_REQ()
 - @c PARAM_DOUBLE_IN_REQ()
 - @c PARAM_STRING_IN_REQ()
 - @c PARAM_MATRIX_IN_REQ()
 - @c PARAM_UMATRIX_IN_REQ()
 - @c PARAM_TMATRIX_IN_REQ()
 - @c PARAM_VECTOR_IN_REQ()
 - @c PARAM_MODEL_IN_REQ()

Click the links for each macro to read further documentation.  Note also that
each possible combination of @c IN, @c OUT, and @c REQ is not available---output
options cannot be required, and some combinations simply have not been added
because they have not been needed.

The @c PARAM_MODEL_IN() and @c PARAM_MODEL_OUT() macros are used to serialize
mlpack models.  These could be used, for instance, to allow the user to save a
trained model (like a linear regression model) or load an input model.  The
first parameter to the @c PARAM_MODEL_IN() or @c PARAM_MODEL_OUT() macro should
be the C++ type of the model to be serialized; this type @b must have a function
<tt>template<typename Archive> void Serialize(Archive&, const unsigned int)</tt>
(i.e. the type must be serializable via mlpack's boost::serialization shim).
For example, to allow a user to specify an input model of type
`LinearRegression`, the follow definition could be used:

@code
PARAM_MODEL_IN(LinearRegression, "input_model", "The input model to be used.",
    "i");
@endcode

Then, the user will be able to specify their model from the command-line as
@c \--input_model_file and from Python using the @c input_model option to the
generated binding.

From the command line, matrix-type and model-type options (both input and
output) are loaded from or saved to the specified file.  This means that
@c _file is appended to the name of the parameter; so if the parameter name is
@c data and it is of a matrix or model type, then the name that the user will
specify on the command line will be @c \--data_file.  This displayed parameter
name change @b only occurs with matrix and model type parameters for
command-line programs.

The @c PARAM_MATRIX_AND_INFO() macro defines a categorical matrix parameter
(more specifically, a matrix type that can support categorical columns).  From
the C++ program side, this means that the parameter type is
<tt>std::tuple<data::DatasetInfo, arma::mat></tt>.  From the user side, for a
command-line program, this means that the user will pass the filename of a
dataset that can have categorical features, such as an ARFF dataset.  For a
Python program, the user may pass a Pandas matrix with categorical columns.
When the program is run, the input that the user gives will be processed and the
@c data::DatasetInfo object will be filled with the dimension types and the
@c arma::mat object will be filled with the data itself.

To give some examples, the parameter definitions from the example
"random_numbers" program in the previous section are shown below.

@code
PARAM_MATRIX_IN("input", "The input matrix that will be ignored.", "i");
PARAM_DOUBLE_IN("subtract", "The value to subtract from each parameter.", "s",
    0.0); // Default value of 0.0.
PARAM_INT_IN("num_samples", "The number of samples to generate.", "n", 100);

PARAM_MATRIX_OUT("output", "The output matrix of random samples.", "o");
PARAM_MODEL_OUT(LinearRegression, "output_model", "The randomly generated "
    "linear regression output model.", "M");
@endcode

Note that even the parameter documentation strings must be a little be agnostic
to the binding type, because the command-line interface is so different than the
Python interface to the user.

@subsection bindings_general_functions Using IO in an mlpackMain() function

mlpack's @c IO module provides a unified abstract interface for getting input
from and providing output to users without needing to consider the language
(command-line, Python, MATLAB, etc.) that the user is running the program from.
This means that after the @c BINDING_LONG_DESC() and @c BINDING_EXAMPLE() macros
and the @c PARAM_*() macros have been defined, a language-agnostic
@c mlpackMain() function can be written. This function then can perform the
actual computation that the entire program is meant to.

Inside of an @c mlpackMain() function, the @c mlpack::IO module can be used to
access input parameters and set output parameters.  There are two main functions
for this, plus a utility printing function:

 - @c IO::GetParam<T>() - get a reference to a parameter
 - @c IO::HasParam() - returns true if the user specified the parameter
 - @c IO::GetPrintableParam<T>() - returns a string representing the value of
      the parameter

So, to print "hello" if the user specified the @c print_hello parameter, the
following code could be used:

@code
using namespace mlpack;

if (IO::HasParam("print_hello"))
  std::cout << "Hello!" << std::endl;
else
  std::cout << "No greetings for you!" << std::endl;
@endcode

To access a string that a user passed in to the @c string parameter, the
following code could be used:

@code
using namespace mlpack;

const std::string& str = IO::GetParam<std::string>("string");
@endcode

Matrix types are accessed in the same way:

@code
using namespace mlpack;

arma::mat& matrix = IO::GetParam<arma::mat>("matrix");
@endcode

Similarly, model types can be accessed.  If a @c LinearRegression model was
specified by the user as the parameter @c model, the following code can access
the model:

@code
using namespace mlpack;

LinearRegression& lr = IO::GetParam<LinearRegression>("model");
@endcode

Matrices with categoricals are a little trickier to access since the C++
parameter type is <tt>std::tuple<data::DatasetInfo, arma::mat></tt>.  The
example below creates references to both the @c DatasetInfo and matrix objects,
assuming the user has passed a matrix with categoricals as the @c matrix
parameter.

@code
using namespace mlpack;

typename std::tuple<data::DatasetInfo, arma::mat> TupleType;
data::DatasetInfo& di = std::get<0>(IO::GetParam<TupleType>("matrix"));
arma::mat& matrix = std::get<1>(IO::GetParam<TupleType>("matrix"));
@endcode

These two functions can be used to write an entire program.  The third function,
@c GetPrintableParam(), can be used to help provide useful output in a program.
Typically, this function should be used if you want to provide some kind of
error message about a matrix or model parameter, but want to avoid printing the
matrix itself.  For instance, printing a matrix parameter with
@c GetPrintableParam() will print the filename for a command-line binding or the
size of a matrix for a Python binding.  @c GetPrintableParam() for a model
parameter will print the filename for the model for a command-line binding or
a simple string representing the type of the model for a Python binding.

Putting all of these ideas together, here is the @c mlpackMain() function that
could be created for the "random_numbers" program from earlier sections.

@code
#include <mlpack/core/util/mlpack_main.hpp>

// BINDING_NAME(), BINDING_SHORT_DESC(), BINDING_LONG_DESC() , BINDING_EXAMPLE(),
// BINDING_SEE_ALSO() and PARAM_*() definitions should go here:
// ...

using namespace mlpack;

void mlpackMain()
{
  // If the user passed an input matrix, tell them that we'll be ignoring it.
  if (IO::HasParam("input"))
  {
    // Print the filename the user passed, if a command-line binding, or the
    // size of the matrix passed, if a Python binding.
    Log::Warn << "The input matrix "
        << IO::GetPrintableParam<arma::mat>("input") << " is ignored!"
        << std::endl;
  }

  // Get the number of samples and also the value we should subtract.
  const size_t numSamples = (size_t) IO::GetParam<int>("num_samples");
  const double subtractValue = IO::GetParam<double>("subtract");

  // Create the random matrix (1-dimensional).
  arma::mat output(1, numSamples, arma::fill::randu);
  output -= subtractValue;

  // Save the output matrix if the user wants.
  if (IO::HasParam("output"))
    IO::GetParam<arma::mat>("output") = std::move(output); // Avoid copy.

  // Did the user request a random linear regression model?
  if (IO::HasParam("output_model"))
  {
    LinearRegression lr;
    lr.Parameters().randu(10); // 10-dimensional (arbitrary).
    lr.Lambda() = 0.0;
    lr.Intercept() = false; // No intercept term.

    IO::GetParam<LinearRegression>("output_model") = std::move(lr);
  }
}
@endcode

@subsection bindings_general_more More documentation on using IO

More documentation for the IO module can either be found on the mlpack::IO
documentation page, or by reading the existing mlpack bindings.  These can be
found in the @c src/mlpack/methods/ folders, by finding the @c _main.cpp files.
For instance, @c src/mlpack/methods/neighbor_search/knn_main.cpp is the
k-nearest-neighbor search program definition.

@section bindings_structure Structure of IO module and associated macros

This section describes the internal functionality of the IO module and the
associated macros.  If you are only interested in writing mlpack programs, this
section is probably not worth reading.

There are eight main components involved with mlpack bindings:

 - the IO module, a singleton class that stores parameter information
 - the mlpackMain() function that defines the functionality of the binding
 - the BINDING_NAME() macro that defines the binding name
 - the BINDING_SHORT_DESC() macro that defines the short description
 - the BINDING_LONG_DESC() macro that defines the long description
 - (optional) the BINDING_EXAMPLE() macro that defines example usages
 - (optional) the BINDING_SEE_ALSO() macro that defines "see also" links
 - the PARAM_*() macros that define parameters for the binding

The mlpack::IO module is a singleton class that stores, at runtime, the binding
name, the documentation, and the parameter information and values.  In order to
do this, each parameter and the program documentation must make themselves known
to the IO singleton.  This is accomplished by having the @c BINDING_NAME(),
@c BINDING_SHORT_DESC(), @c BINDING_LONG_DESC(), @c BINDING_EXAMPLE(),
@c BINDING_SEE_ALSO() and @c PARAM_*() macros declare global variables that,
in their constructors, register themselves with the IO singleton.

The @c BINDING_NAME() macro declares an object of type mlpack::util::ProgramName.
The @c BINDING_SHORT_DESC() macro declares an object of type
mlpack::util::ShortDescription.
The @c BINDING_LONG_DESC() macro declares an object of type
mlpack::util::LongDescription.
The @c BINDING_EXAMPLE() macro declares an object of type mlpack::util::Example.
The @c BINDING_SEE_ALSO() macro declares an object of type
mlpack::util::SeeAlso.
The @c ProgramName class constructor calls IO::RegisterProgramName() in order to
register the given program name.
The @c ShortDescription class constructor calls IO::RegisterShortDescription() in order to
register the given short description.
The @c LongDescription class constructor calls IO::RegisterLongDescription() in order to
register the given long description.
The @c Example class constructor calls IO::RegisterExample() in order to
register the given example.
The @c SeeAlso class constructor calls IO::RegisterSeeAlso() in order to
register the given see-also link.

The @c PARAM_*() macros declare an object that will, in its constructor, call
IO::Add() to register that parameter with the IO singleton.  The specific type
of that object will depend on the binding type being used.

The IO::Add() function takes an mlpack::util::ParamData object as its input.
This @c ParamData object has a number of fields that must be set to properly
describe the parameter.  Each of the fields is documented and probably
self-explanatory, but three fields deserve further explanation:

 - the <tt>std::string tname</tt> member is used to encode the true type of
   the parameter---which is not known by the IO singleton at runtime.  This
   should be set to <tt>TYPENAME(T)</tt> where @c T is the type of the
   parameter.

 - the <tt>boost::any value</tt> member is used to hold the actual value of the
   parameter.  Typically this will simply be the parameter held by a
   @c boost::any object, but for some types it may be more complex.  For
   instance, for a command-line matrix option, the @c value parameter will
   actually hold a tuple containing both the filename and the matrix itself.

 - the <tt>std::string cppType</tt> should be a string containing the type as
   seen in C++ code.  Typically this can be encoded by stringifying a
   @c PARAM_*() macro argument.

Thus, the global object defined by the @c PARAM_*() macro must turn its
arguments into a fully specified @c ParamData object and then call IO::Add()
with it.

With different binding types, different behavior is often required for the
@c GetParam<T>(), @c HasParam(), and @c GetPrintableParam<T>() functions.  In
order to handle this, the IO singleton also holds a function pointer map, so
that a given type of option can call specific functionality for a certain task.
This function map is accessible as @c IO::functionMap and is not meant to be
used by users, but instead by people writing binding types.

Each function in the map must have signature

@code
void MapFunction(const util::ParamData& d,
                 const void* input,
                 void* output);
@endcode

The use of void pointers allows any type to be specified as input or output to
the function without changing the signature for the map.  The IO function map
is of type

@code
std::map<std::string, std::map<std::string,
    void (*)(const util::ParamData&, const void*, void*)>>
@endcode

and the first map key is the typename (<tt>tname</tt>) of the parameter, and the
second map key is the string name of the function.  For instance, calling

@code
const util::ParamData& d = IO::Parameters()["param"];
IO::GetSingleton().functionMap[d.tname]["GetParam"](d, input, output);
@endcode

will call the @c GetParam() function for the type of the @c "param" parameter.
Examples are probably easiest to understand how this functionality works; see
the IO::GetParam<T>() source to see how this might be used.

The IO singleton expects the following functions to be defined in the function
map for each type:

 - @c GetParam -- return a pointer to the parameter in @c output.
 - @c GetPrintableParam -- return a pointer to a string description of the
       parameter in @c output.

If these functions are properly defined, then the IO module will work
correctly.  Other functions may also be defined; these may be used by other
parts of the binding infrastructure for different languages.

@section bindings_cli Command-line program bindings

This section describes the internal functionality of the command-line program
binding generator.  If you are only interested in writing mlpack programs, this
section probably is not worth reading.  This section is worth reading only if
you want to know the specifics of how the @c mlpackMain() function and macros
get turned into a fully working command-line program.

The code for the command-line bindings is found in @c src/mlpack/bindings/cli.

@subsection bindings_cli_mlpack_main mlpackMain() definition

Any command-line program must be compiled with the @c BINDING_TYPE macro
set to the value @c BINDING_TYPE_CLI.  This is handled by the CMake macro
@c add_cli_executable().

When @c BINDING_TYPE is set to @c BINDING_TYPE_CLI, the following is set in
@c src/mlpack/core/util/mlpack_main.hpp, which must be included by every mlpack
binding:

 - The options defined by @c PARAM_*() macros are of type
   mlpack::bindings::cli::CLIOption.

 - The parameter and value printing macros for @c BINDING_LONG_DESC()
   and BINDING_EXAMPLE() are set:
   * The @c PRINT_PARAM_STRING() macro is defined as
     mlpack::bindings::cli::ParamString().
   * The @c PRINT_DATASET() macro is defined as
     mlpack::bindings::cli::PrintDataset().
   * The @c PRINT_MODEL() macro is defined as
     mlpack::bindings::cli::PrintModel().
   * The @c PRINT_CALL() macro is defined as
     mlpack::bindings::cli::ProgramCall().

 - The function <tt>int main()</tt> is defined as:

@code
int main(int argc, char** argv)
{
  // Parse the command-line options; put them into IO.
  mlpack::bindings::cli::ParseCommandLine(argc, argv);

  mlpackMain();

  // Print output options, print verbose information, save model parameters,
  // clean up, and so forth.
  mlpack::bindings::cli::EndProgram();
}
@endcode

Thus any mlpack command-line binding first processes the command-line arguments
with mlpack::bindings::cli::ParseCommandLine(), then runs the binding with
@c mlpackMain(), then cleans up with mlpack::bindings::cli::EndProgram().

The @c ParseCommandLine() function reads the input parameters and sets the
values in IO.  For matrix-type and model-type parameters, this reads the
filenames from the command-line, but does not load the matrix or model.  Instead
the matrix or model is loaded the first time it is accessed with
@c GetParam<T>().

The @c \--help parameter is handled by the mlpack::bindings::cli::PrintHelp()
function.

At the end of program execution, the mlpack::bindings::cli::EndProgram()
function is called.  This writes any output matrix or model parameters to disk,
and prints the program parameters and timers if @c \--verbose was given.

@subsection bindings_cli_matrix Matrix and model parameter handling

For command line bindings, the matrix, model, and matrix with categorical type
parameters all require special handling, since it is not possible to pass a
matrix of any reasonable size or a model on the command line directly.
Therefore for a matrix or model parameter, the user specifies the file
containing that matrix or model parameter.  If the parameter is an input
parameter, then the file is loaded when @c GetParam<T>() is called.  If the
parameter is an output parameter, then the matrix or model is saved to the file
when @c EndProgram() is called.

The actual implementation of this is that the <tt>boost::any value</tt> member
of the @c ParamData struct does not hold the model or the matrix, but instead a
<tt>std::tuple</tt> containing both the matrix or the model, and the filename
associated with that matrix or model.

This means that functions like @c GetParam<T>() and @c GetPrintableParam<T>()
(and all of the other associated functions in the IO function map) must have
special handling for matrix or model types.  See those implementatipns for more
details---the special handling is enforced via SFINAE.

@subsection bindings_cli_parsing Parsing the command line

The @c ParseCommandLine() function uses <tt>CLI11</tt> to read
the values from the command line into the @c ParamData structs held by the IO
singleton.

In order to set up <tt>CLI11</tt>---and to keep its headers
from needing to be included by the rest of the library---the code loops over
each parameter known by the IO singleton and calls the @c "AddToPO" function
from the function map.  This in turn calls the necessary functions to register a
given parameter with <tt>CLI11</tt>, and once all parameters
have been registered, the facilities provided by <tt>CLI11</tt>
are used to parse the command line input properly.

@section bindings_python Python bindings

This section describes the internal functionality of the mlpack Python binding
generator.  If you are only interested in writing new bindings or building the
bindings, this section is probably not worth reading.  But if you are interested
in the internal working of the Python binding generator, then this section is
for you.

The Python bindings are significantly more complex than the command line
bindings because we cannot just compile directly to a finished product.  Instead
we need a multi-stage compilation:

 - We must generate a setup.py file that can be used to compile the bindings.
 - We must generate the .pyx (Cython) bindings for each program.
 - Then we must build each .pyx into a .so that is loadable from Python.
 - We must also test the Python bindings.

This is done with a combination of C++ code to generate the .pyx bindings, CMake
to run the actual compilation and generate the setup.py file, some utility
Python functions, and tests written in both Python and C++.  This code is
primarily contained in @c src/mlpack/bindings/python/.

@subsection bindings_python_matrix Passing matrices to/from Python

The standard Python matrix library is numpy, so mlpack bindings should accept
numpy matrices as input.  Fortunately, numpy Cython bindings already exist,
which make it easy to convert from a numpy object to an Armadillo object without
copying any data.  This code can be found in
@c src/mlpack/bindings/python/mlpack/arma_numpy.pyx, and is used by the Python
@c GetParam<T>() functionality.

mlpack also supports categorical matrices; in Python, the typical way of
representing matrices with categorical features is with Pandas.  Therefore,
mlpack also accepts Pandas matrices, and if any of the Pandas matrix dimensions
are categorical, these are properly encoded.  The function
@c to_matrix_with_info() from @c mlpack/bindings/python/mlpack/matrix_utils.py
is used to perform this conversion.

@subsection bindings_python_model Passing model parameter to/from Python

We use (or abuse) Cython functionality in order to give the user a model object
that they can use in their Python code.  However, we do not want to (or have the
infrastructure to) write bindings for every method that a serializable model
class might support; therefore, we only desire to return a memory pointer to the
model to the user.

In this way, a user that receives a model from an output parameter can then
reuse the model as an input parameter to another binding (or the same binding).

To return a function pointer we have to define a Cython class in the following
way (this example is taken from the perceptron binding):

@code
cdef extern from "</home/ryan/src/mlpack-rc/src/mlpack/methods/perceptron/perceptron_main.cpp>" nogil:
  cdef int mlpackMain() nogil except +RuntimeError

  cdef cppclass PerceptronModel:
    PerceptronModel() nogil


cdef class PerceptronModelType:
  cdef PerceptronModel* modelptr

  def __cinit__(self):
    self.modelptr = new PerceptronModel()

  def __dealloc__(self):
    del self.modelptr
@endcode

This class definition is automatically generated when the .pyx file is
automatically generated.

@subsection bindings_python_setup_py CMake generation of setup.py

A boilerplate setup.py file can be found in
@c src/mlpack/bindings/python/setup.py.in.  This will be configured by CMake to
produce the final @c setup.py file, but in order to do this, a list of the .pyx
files to be compiled must be gathered.

Therefore, the @c add_python_binding() macro is defined in
@c src/mlpack/bindings/python/CMakeLists.txt.  This adds the given binding to
the @c MLPACK_PYXS variable, which is then inserted into @c setup.py as part of
the @c configure_file() step in @c src/mlpack/CMakeLists.txt.

@subsection bindings_python_generate_pyx Generation of .pyx files

A binding named @c program is built into a program called
@c generate_pyx_program (this a CMake target, so you can build these
individually if you like).  The file
@c src/mlpack/bindings/python/generate_pyx.cpp.in is configured by CMake to set
the name of the program and the @c *_main.cpp file to include correctly, then
the @c mlpack::bindings::python::PrintPYX() function is called by the program.
The @c PrintPYX() function uses the parameters that have been set in the IO
singleton by the @c BINDING_NAME(), @c BINDING_SHORT_DESC(),
@c BINDING_LONG_DESC(), @c BINDING_EXAMPLE(), @c BINDING_SEE_ALSO() and
@c PARAM_*() macros in order to actually print a fully-working .pyx file that
can be compiled.  The file has several sections:

 - Python imports (numpy/pandas/cython/etc.)
 - Cython imports of C++ utility functions and Armadillo functionality
 - Cython imports of any necessary serializable model types
 - Definitions of classes for serializable model types
 - The binding function definition
 - Documentation: input and output parameters
 - The call to mlpackMain()
 - Handling of output functionality
 - Return of output parameters

Any output parameters for Python bindings are returned in a dict containing
named elements.

@subsection bindings_python_build_pyx Building the .pyx files

After building the @c generate_pyx_program target, the @c build_pyx_program
target is built as a dependency of the @c python target.  This simply takes the
generated .pyx file and uses Python setuptools to compile this to a Python
binding.

@subsection bindings_python_testing Testing the Python bindings

We cannot do our tests only from the Boost Unit Test Framework in C++ because we
need to see that we are able to load parameters properly from Python and return
output correctly.

The tests are in @c src/mlpack/bindings/python/tests/ and test both the actual
bindings and also the auxiliary Python code included in
@c src/mlpack/bindings/python/mlpack/.

@section bindings_new Adding new binding types

Adding a new binding type to mlpack is fairly straightforward once the general
structure of the IO singleton and the function map that IO uses is understood.
For each different language that bindings are desired for, the route to a
solution will be particularly different---so it is hard to provide any general
guidance for how to make new bindings that will be applicable to each language.

In general, the first thing to handle will be how matrices are passed back and
forth between the target language.  Typically this might mean getting the memory
address of an input matrix and wrapping an @c arma::mat object around that
memory address.  This can be handled in the @c GetParam() function that is part
of the IO singleton function map; see @c get_param.hpp for both the IO and
Python bindings for an example (in @c src/mlpack/bindings/cli/ and
@c src/mlpack/bindings/python/).

Serialization of models is also a tricky consideration; in some languages you
will be able to pass a pointer to the model itself.  This is generally
best---users should not expect to be able to manipulate the model in the target
language, but they should expect that they can pass a model back and forth
without paying a runtime penalty.  So, for example, serializing a model using a
@c boost::text_oarchive and then returning the string that represents the model
is not acceptable, because that string can be extremely large and the time it
takes to decode the model can be very large.

The strategy of generating a binding definition for the target language, like
what is done with Python, can be a useful strategy that should be considered.
If this is the route that is desired, a large amount of CMake boilerplate may be
necessary.  The Python CMake configuration can be referred to as an example, but
probably a large amount of adaptation to other languages will be necessary.

Lastly, when adding a new language, be sure to make sure it works with the
Markdown documentation generator.  In order to make this happen, you will need
to modify all of the @c add_markdown_docs() calls in the different
@c CMakeLists.txt files to contain the name of the language you have written a
binding for.  You will also need to modify every function in
@c src/mlpack/bindings/markdown/print_doc_functions_impl.hpp to correctly call
out to the corresponding function for the language that you have written
bindings for.

*/
