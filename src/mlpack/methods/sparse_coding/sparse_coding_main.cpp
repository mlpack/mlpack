/**
 * @file methods/sparse_coding/sparse_coding_main.cpp
 * @author Nishant Mehta
 *
 * Executable for Sparse Coding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME sparse_coding

#include <mlpack/core/util/mlpack_main.hpp>

#include "sparse_coding.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Sparse Coding");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Sparse Coding with Dictionary Learning.  Given a "
    "dataset, this will decompose the dataset into a sparse combination of a "
    "few dictionary elements, where the dictionary is learned during "
    "computation; a dictionary can be reused for future sparse coding of new "
    "points.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of Sparse Coding with Dictionary Learning, which "
    "achieves sparsity via an l1-norm regularizer on the codes (LASSO) or an "
    "(l1+l2)-norm regularizer on the codes (the Elastic Net).  Given a dense "
    "data matrix X with d dimensions and n points, sparse coding seeks to find "
    "a dense dictionary matrix D with k atoms in d dimensions, and a sparse "
    "coding matrix Z with n points in k dimensions."
    "\n\n"
    "The original data matrix X can then be reconstructed as Z * D.  Therefore,"
    " this program finds a representation of each point in X as a sparse linear"
    " combination of atoms in the dictionary D."
    "\n\n"
    "The sparse coding is found with an algorithm which alternates between a "
    "dictionary step, which updates the dictionary D, and a sparse coding step,"
    " which updates the sparse coding matrix."
    "\n\n"
    "Once a dictionary D is found, the sparse coding model may be used to "
    "encode other matrices, and saved for future usage."
    "\n\n"
    "To run this program, either an input matrix or an already-saved sparse "
    "coding model must be specified.  An input matrix may be specified with the"
    " " + PRINT_PARAM_STRING("training") + " option, along with the number of "
    "atoms in the dictionary (specified with the " +
    PRINT_PARAM_STRING("atoms") + " parameter).  It is also possible to specify"
    " an initial dictionary for the optimization, with the " +
    PRINT_PARAM_STRING("initial_dictionary") + " parameter.  An input model may"
    " be specified with the " + PRINT_PARAM_STRING("input_model") +
    " parameter.");

// Example.
BINDING_EXAMPLE(
    "As an example, to build a sparse coding model on the dataset " +
    PRINT_DATASET("data") + " using 200 atoms and an l1-regularization "
    "parameter of 0.1, saving the model into " + PRINT_MODEL("model") + ", use "
    "\n\n" +
    PRINT_CALL("sparse_coding", "training", "data", "atoms", 200, "lambda1",
        0.1, "output_model", "model") +
    "\n\n"
    "Then, this model could be used to encode a new matrix, " +
    PRINT_DATASET("otherdata") + ", and save the output codes to " +
    PRINT_DATASET("codes") + ": "
    "\n\n" +
    PRINT_CALL("sparse_coding", "input_model", "model", "test", "otherdata",
        "codes", "codes"));

// See also...
BINDING_SEE_ALSO("@local_coordinate_coding", "#local_coordinate_coding");
BINDING_SEE_ALSO("Sparse dictionary learning on Wikipedia",
    "https://en.wikipedia.org/wiki/Sparse_dictionary_learning");
BINDING_SEE_ALSO("Efficient sparse coding algorithms (pdf)",
    "https://proceedings.neurips.cc/paper_files/paper/2006/file/"
    "2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf");
BINDING_SEE_ALSO("Regularization and variable selection via the elastic net",
    "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf"
    "&doi=46217f372a75dddc2254fdbc6b9418ba3554e453");
BINDING_SEE_ALSO("SparseCoding C++ class documentation",
    "@doc/user/methods/sparse_coding.md");

// Train the model.
PARAM_MATRIX_IN("training", "Matrix of training data (X).", "t");
PARAM_INT_IN("atoms", "Number of atoms in the dictionary.", "k", 15);

PARAM_DOUBLE_IN("lambda1", "Sparse coding l1-norm regularization parameter.",
    "l", 0);
PARAM_DOUBLE_IN("lambda2", "Sparse coding l2-norm regularization parameter.",
    "L", 0);
PARAM_INT_IN("max_iterations", "Maximum number of iterations for sparse coding "
    "(0 indicates no limit).", "n", 0);
PARAM_MATRIX_IN("initial_dictionary", "Optional initial dictionary matrix.",
    "i");
PARAM_FLAG("normalize", "If set, the input data matrix will be normalized "
    "before coding.", "N");
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_DOUBLE_IN("objective_tolerance", "Tolerance for convergence of the "
    "objective function.", "o", 0.01);
PARAM_DOUBLE_IN("newton_tolerance", "Tolerance for convergence of Newton "
    "method.", "w", 1e-6);

// Load/save a model.
PARAM_MODEL_IN(SparseCoding<>, "input_model", "File containing input sparse "
    "coding model.", "m");
PARAM_MODEL_OUT(SparseCoding<>, "output_model", "File to save trained sparse "
    "coding model to.", "M");

PARAM_MATRIX_OUT("dictionary", "Matrix to save the output dictionary to.", "d");
PARAM_MATRIX_OUT("codes", "Matrix to save the output sparse codes of the test "
    "matrix (--test_file) to.", "c");

PARAM_MATRIX_IN("test", "Optional matrix to be encoded by trained model.", "T");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Check for parameter validity.
  if (params.Has("input_model") && params.Has("initial_dictionary"))
  {
    Log::Fatal << "Can only pass one of " << PRINT_PARAM_STRING("input_model")
        << " or " << PRINT_PARAM_STRING("initial_dictionary") << "!" << endl;
  }

  if (params.Has("training"))
  {
    RequireAtLeastOnePassed(params, { "atoms" }, true, "if training data is "
        "specified, the number of atoms in the dictionary must also be "
        "specified");
  }

  RequireAtLeastOnePassed(params, { "codes", "dictionary", "output_model" },
      false, "no output will be saved");

  ReportIgnoredParam(params, {{ "test", false }}, "codes");

  ReportIgnoredParam(params, {{ "training", false }}, "atoms");
  ReportIgnoredParam(params, {{ "training", false }}, "lambda1");
  ReportIgnoredParam(params, {{ "training", false }}, "lambda2");
  ReportIgnoredParam(params, {{ "training", false }}, "initial_dictionary");
  ReportIgnoredParam(params, {{ "training", false }}, "max_iterations");
  ReportIgnoredParam(params, {{ "training", false }}, "normalize");
  ReportIgnoredParam(params, {{ "training", false }}, "objective_tolerance");
  ReportIgnoredParam(params, {{ "training", false }}, "newton_tolerance");

  RequireParamValue<int>(params, "atoms", [](int x) { return x > 0; }, true,
      "number of atoms must be positive");
  RequireParamValue<double>(params, "lambda1",
      [](double x) { return x >= 0.0; }, true, "lambda1 value must be "
      "nonnegative");
  RequireParamValue<double>(params, "lambda2",
      [](double x) { return x >= 0.0; }, true, "lambda2 value must be "
      "nonnegative");
  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum number of iterations must be nonnegative");
  RequireParamValue<double>(params, "objective_tolerance",
      [](double x) { return x >= 0.0; }, true,
      "objective function tolerance must be nonnegative");
  RequireParamValue<double>(params, "newton_tolerance",
      [](double x) { return x >= 0.0; }, true,
      "Newton method tolerance must be nonnegative");

  // Do we have an existing model?
  SparseCoding<>* sc;
  if (params.Has("input_model"))
    sc = params.Get<SparseCoding<>*>("input_model");
  else
    sc = new SparseCoding<>(0, 0.0);

  if (params.Has("training"))
  {
    mat matX = std::move(params.Get<arma::mat>("training"));

    // Normalize each point if the user asked for it.
    if (params.Has("normalize"))
    {
      Log::Info << "Normalizing data before coding..." << endl;
      for (size_t i = 0; i < matX.n_cols; ++i)
        matX.col(i) /= norm(matX.col(i), 2);
    }

    sc->Lambda1() = params.Get<double>("lambda1");
    sc->Lambda2() = params.Get<double>("lambda2");
    sc->MaxIterations() = (size_t) params.Get<int>("max_iterations");
    sc->Atoms() = (size_t) params.Get<int>("atoms");
    sc->ObjTolerance() = params.Get<double>("objective_tolerance");
    sc->NewtonTolerance() = params.Get<double>("newton_tolerance");

    // Inform the user if we are overwriting their model.
    timers.Start("sparse_coding");
    if (params.Has("input_model"))
    {
      Log::Info << "Using dictionary from existing model in '"
          << params.GetPrintable<SparseCoding<>>("input_model")
          << "' as initial dictionary for training." << endl;
      sc->Train<NothingInitializer>(matX);
    }
    else if (params.Has("initial_dictionary"))
    {
      // Load initial dictionary directly into sparse coding object.
      sc->Dictionary() =
          std::move(params.Get<arma::mat>("initial_dictionary"));

      // Validate size of initial dictionary.
      if (sc->Dictionary().n_cols != sc->Atoms())
      {
        const size_t dictAtoms = sc->Dictionary().n_cols;
        const size_t atoms = sc->Atoms();
        if (!params.Has("input_model"))
          delete sc;
        Log::Fatal << "The initial dictionary has " << dictAtoms
            << " atoms, but the number of atoms was specified to be "
            << atoms << "!" << endl;
      }

      if (sc->Dictionary().n_rows != matX.n_rows)
      {
        const size_t dim = sc->Dictionary().n_rows;
        if (!params.Has("input_model"))
          delete sc;
        Log::Fatal << "The initial dictionary has " << dim
            << " dimensions, but the data has " << matX.n_rows << " dimensions!"
            << endl;
      }

      // Run sparse coding.
      sc->Train<NothingInitializer>(matX);
    }
    else
    {
      // Run sparse coding with the default initialization.
      sc->Train(matX);
    }
    timers.Stop("sparse_coding");
  }

  // Now, de we have any matrix to encode?
  if (params.Has("test"))
  {
    if (params.Get<arma::mat>("test").n_rows != sc->Dictionary().n_rows)
    {
      const size_t dim = sc->Dictionary().n_rows;
      if (!params.Has("input_model"))
        delete sc;
      Log::Fatal << "Model was trained with a dimensionality of "
          << dim << ", but test data '"
          << params.GetPrintable<arma::mat>("test") << "' have a "
          << "dimensionality of " << params.Get<arma::mat>("test").n_rows
          << "!" << endl;
    }

    mat matY = std::move(params.Get<arma::mat>("test"));

    // Normalize each point if the user asked for it.
    if (params.Has("normalize"))
    {
      Log::Info << "Normalizing test data before coding..." << endl;
      for (size_t i = 0; i < matY.n_cols; ++i)
        matY.col(i) /= norm(matY.col(i), 2);
    }

    mat codes;
    sc->Encode(matY, codes);

    params.Get<arma::mat>("codes") = std::move(codes);
  }

  // Did the user want to save the dictionary?  Use an alias for the dictionary.
  params.Get<arma::mat>("dictionary") = arma::mat(sc->Dictionary().memptr(),
      sc->Dictionary().n_rows, sc->Dictionary().n_cols, false, false);

  // Save the model.
  params.Get<SparseCoding<>*>("output_model") = sc;
}
