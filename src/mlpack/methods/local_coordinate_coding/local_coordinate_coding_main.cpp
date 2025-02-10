/**
 * @file methods/local_coordinate_coding/local_coordinate_coding_main.cpp
 * @author Nishant Mehta
 *
 * Executable for Local Coordinate Coding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME local_coordinate_coding

#include <mlpack/core/util/mlpack_main.hpp>

#include "lcc.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Local Coordinate Coding");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Local Coordinate Coding (LCC), a data transformation "
    "technique.  Given input data, this transforms each point to be expressed "
    "as a linear combination of a few points in the dataset; once an LCC model "
    "is trained, it can be used to transform points later also.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of Local Coordinate Coding (LCC), which "
    "codes data that approximately lives on a manifold using a variation of l1-"
    "norm regularized sparse coding.  Given a dense data matrix X with n points"
    " and d dimensions, LCC seeks to find a dense dictionary matrix D with k "
    "atoms in d dimensions, and a coding matrix Z with n points in k "
    "dimensions.  Because of the regularization method used, the atoms in D "
    "should lie close to the manifold on which the data points lie."
    "\n\n"
    "The original data matrix X can then be reconstructed as D * Z.  Therefore,"
    " this program finds a representation of each point in X as a sparse linear"
    " combination of atoms in the dictionary D."
    "\n\n"
    "The coding is found with an algorithm which alternates between a "
    "dictionary step, which updates the dictionary D, and a coding step, which "
    "updates the coding matrix Z."
    "\n\n"
    "To run this program, the input matrix X must be specified (with -i), along"
    " with the number of atoms in the dictionary (-k).  An initial dictionary "
    "may also be specified with the " +
    PRINT_PARAM_STRING("initial_dictionary") + " parameter.  The l1-norm "
    "regularization parameter is specified with the " +
    PRINT_PARAM_STRING("lambda") + " parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to run LCC on "
    "the dataset " + PRINT_DATASET("data") + " using 200 atoms and an "
    "l1-regularization parameter of 0.1, saving the dictionary " +
    PRINT_PARAM_STRING("dictionary") + " and the codes into " +
    PRINT_PARAM_STRING("codes") + ", use"
    "\n\n" +
    PRINT_CALL("local_coordinate_coding", "training", "data", "atoms", 200,
        "lambda", 0.1, "dictionary", "dict", "codes", "codes") +
    "\n\n"
    "The maximum number of iterations may be specified with the " +
    PRINT_PARAM_STRING("max_iterations") + " parameter. "
    "Optionally, the input data matrix X can be normalized before coding with "
    "the " + PRINT_PARAM_STRING("normalize") + " parameter."
    "\n\n"
    "An LCC model may be saved using the " +
    PRINT_PARAM_STRING("output_model") + " output parameter.  Then, to encode "
    "new points from the dataset " + PRINT_DATASET("points") + " with the "
    "previously saved model " + PRINT_MODEL("lcc_model") + ", saving the new "
    "codes to " + PRINT_DATASET("new_codes") + ", the following command can "
    "be used:"
    "\n\n" +
    PRINT_CALL("local_coordinate_coding", "input_model", "lcc_model", "test",
        "points", "codes", "new_codes"));

// See also...
BINDING_SEE_ALSO("@sparse_coding", "#sparse_coding");
BINDING_SEE_ALSO("Nonlinear learning using local coordinate coding (pdf)",
    "https://proceedings.neurips.cc/paper_files/paper/2009/file/"
    "2afe4567e1bf64d32a5527244d104cea-Paper.pdf");
BINDING_SEE_ALSO("LocalCoordinateCoding C++ class documentation",
    "@doc/user/methods/local_coordinate_coding.md");

// Training parameters.
PARAM_MATRIX_IN("training", "Matrix of training data (X).", "t");
PARAM_INT_IN("atoms", "Number of atoms in the dictionary.", "k", 0);
PARAM_DOUBLE_IN("lambda", "Weighted l1-norm regularization parameter.", "l",
    0.0);
PARAM_INT_IN("max_iterations", "Maximum number of iterations for LCC (0 "
    "indicates no limit).", "n", 0);
PARAM_MATRIX_IN("initial_dictionary", "Optional initial dictionary.", "i");
PARAM_FLAG("normalize", "If set, the input data matrix will be normalized "
    "before coding.", "N");
PARAM_DOUBLE_IN("tolerance", "Tolerance for objective function.", "o", 0.01);

// Load/save a model.
PARAM_MODEL_IN(LocalCoordinateCoding<>, "input_model", "Input LCC model.", "m");
PARAM_MODEL_OUT(LocalCoordinateCoding<>, "output_model",
    "Output for trained LCC model.", "M");

// Test on another dataset.
PARAM_MATRIX_IN("test", "Test points to encode.", "T");
PARAM_MATRIX_OUT("dictionary", "Output dictionary matrix.", "d");
PARAM_MATRIX_OUT("codes", "Output codes matrix.", "c");

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // Check for parameter validity.
  RequireOnlyOnePassed(params, { "training", "input_model" }, true);

  if (params.Has("training"))
    RequireAtLeastOnePassed(params, { "atoms" }, true);

  RequireAtLeastOnePassed(params, { "codes", "dictionary", "output_model" },
      false, "no output will be saved");

  ReportIgnoredParam(params, {{ "test", false }}, "codes");

  ReportIgnoredParam(params, {{ "training", false }}, "atoms");
  ReportIgnoredParam(params, {{ "training", false }}, "lambda");
  ReportIgnoredParam(params, {{ "training", false }}, "initial_dictionary");
  ReportIgnoredParam(params, {{ "training", false }}, "max_iterations");
  ReportIgnoredParam(params, {{ "training", false }}, "normalize");
  ReportIgnoredParam(params, {{ "training", false }}, "tolerance");

  // Do we have an existing model?
  LocalCoordinateCoding<>* lcc = NULL;
  if (params.Has("input_model"))
    lcc = params.Get<LocalCoordinateCoding<>*>("input_model");

  if (params.Has("training"))
  {
    mat matX = std::move(params.Get<mat>("training"));

    // Normalize each point if the user asked for it.
    if (params.Has("normalize"))
    {
      Log::Info << "Normalizing data before coding..." << endl;
      for (size_t i = 0; i < matX.n_cols; ++i)
        matX.col(i) /= norm(matX.col(i), 2);
    }

    // Check if the parameters lie within the bounds.
    RequireParamValue<int>(params, "atoms", [&matX](int x)
        { return (x > 0) && ((size_t) x < matX.n_cols); }, 1,
        "Number of atoms must lie between 1 and number of training points");

    RequireParamValue<double>(params, "lambda", [](double x) { return x >= 0; },
        1, "The regularization parameter should be a non-negative real number");

    RequireParamValue<double>(params, "tolerance",
        [](double x) { return x > 0; }, 1,
        "Tolerance should be a positive real number");

    lcc = new LocalCoordinateCoding<>(0, 0.0);

    lcc->Lambda() = params.Get<double>("lambda");
    lcc->Atoms() = (size_t) params.Get<int>("atoms");
    lcc->MaxIterations() = (size_t) params.Get<int>("max_iterations");
    lcc->Tolerance() = params.Get<double>("tolerance");

    // Inform the user if we are overwriting their model.
    timers.Start("local_coordinate_coding");
    if (params.Has("input_model"))
    {
      Log::Info << "Using dictionary from existing model in '"
          << params.GetPrintable<string>("input_model") << "' as initial "
          << "dictionary for training." << endl;
      lcc->Train<NothingInitializer>(matX);
    }
    else if (params.Has("initial_dictionary"))
    {
      // Load initial dictionary directly into LCC object.
      lcc->Dictionary() = std::move(params.Get<mat>("initial_dictionary"));

      // Validate the size of the initial dictionary.
      if (lcc->Dictionary().n_cols != lcc->Atoms())
      {
        const size_t dictionarySize = lcc->Dictionary().n_cols;
        const size_t atoms = lcc->Atoms();
        if (!params.Has("input_model"))
          delete lcc;
        Log::Fatal << "The initial dictionary has " << dictionarySize
            << " atoms, but the number of atoms was specified to be "
            << atoms << "!" << endl;
      }

      if (lcc->Dictionary().n_rows != matX.n_rows)
      {
        const size_t dictionaryDimension = lcc->Dictionary().n_rows;
        if (!params.Has("input_model"))
          delete lcc;
        Log::Fatal << "The initial dictionary has " << dictionaryDimension
            << " dimensions, but the data has " << matX.n_rows << " dimensions!"
            << endl;
      }

      // Train the model.
      lcc->Train<NothingInitializer>(matX);
    }
    else
    {
      // Run with the default initialization.
      lcc->Train(matX);
    }
    timers.Stop("local_coordinate_coding");
  }

  // Now, do we have any matrix to encode?
  if (params.Has("test"))
  {
    if (params.Get<mat>("test").n_rows != lcc->Dictionary().n_rows)
    {
      const size_t dictionaryDimension = lcc->Dictionary().n_rows;
      if (!params.Has("input_model"))
        delete lcc;
      Log::Fatal << "Model was trained with a dimensionality of "
          << dictionaryDimension << ", but data in test file "
          << params.GetPrintable<mat>("test") << " has a dimensionality of "
          << params.Get<mat>("test").n_rows << "!" << endl;
    }

    mat matY = std::move(params.Get<mat>("test"));

    // Normalize each point if the user asked for it.
    if (params.Has("normalize"))
    {
      Log::Info << "Normalizing test data before coding..." << endl;
      for (size_t i = 0; i < matY.n_cols; ++i)
        matY.col(i) /= norm(matY.col(i), 2);
    }

    mat codes;
    lcc->Encode(matY, codes);

    params.Get<mat>("codes") = std::move(codes);
  }

  // Save the dictionary and the model.
  params.Get<mat>("dictionary") = lcc->Dictionary();
  params.Get<LocalCoordinateCoding<>*>("output_model") = lcc;
}
