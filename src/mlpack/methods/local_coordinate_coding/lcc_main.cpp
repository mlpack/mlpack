/**
 * @file lcc_main.cpp
 * @author Nishant Mehta
 *
 * Executable for Local Coordinate Coding.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "lcc.hpp"

PROGRAM_INFO("Local Coordinate Coding",
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
    "may also be specified with the --initial_dictionary option.  The l1-norm "
    "regularization parameter is specified with -l.  For example, to run LCC on"
    " the dataset in data.csv using 200 atoms and an l1-regularization "
    "parameter of 0.1, saving the dictionary into dict.csv and the codes into "
    "codes.csv, use "
    "\n\n"
    "$ local_coordinate_coding -i data.csv -k 200 -l 0.1 -d dict.csv -c "
    "codes.csv"
    "\n\n"
    "The maximum number of iterations may be specified with the -n option. "
    "Optionally, the input data matrix X can be normalized before coding with "
    "the -N option.");

// Training parameters.
PARAM_STRING("training_file", "Filename of the training data (X).", "t", "");
PARAM_INT("atoms", "Number of atoms in the dictionary.", "k", 0);
PARAM_DOUBLE("lambda", "Weighted l1-norm regularization parameter.", "l", 0.0);
PARAM_INT("max_iterations", "Maximum number of iterations for LCC (0 indicates "
    "no limit).", "n", 0);
PARAM_STRING("initial_dictionary", "Filename for optional initial dictionary.",
    "i", "");
PARAM_FLAG("normalize", "If set, the input data matrix will be normalized "
    "before coding.", "N");
PARAM_DOUBLE("tolerance", "Tolerance for objective function.", "o", 0.01);

// Load/save a model.
PARAM_STRING("input_model_file", "File containing input LCC model.", "m", "");
PARAM_STRING("output_model_file", "File to save trained LCC model to.", "M",
    "");

// Test on another dataset.
PARAM_STRING("test_file", "File of test points to encode.", "T", "");
PARAM_STRING("dictionary_file", "Filename to save the output dictionary to.",
    "d", "");
PARAM_STRING("codes_file", "Filename to save the output codes to.", "c", "");

PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::lcc;
using namespace mlpack::sparse_coding; // For NothingInitializer.

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // Check for parameter validity.
  if (CLI::HasParam("input_model_file") && CLI::HasParam("initial_dictionary"))
    Log::Fatal << "Cannot specify both --input_model_file (-m) and "
        << "--initial_dictionary (-i)!" << endl;

  if (CLI::HasParam("training_file") && !CLI::HasParam("atoms"))
    Log::Fatal << "If --training_file is specified, the number of atoms in the "
        << "dictionary must be specified with --atoms (-k)!" << endl;

  if (!CLI::HasParam("training_file") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "One of --training_file (-t) or --input_model_file (-m) must "
        << "be specified!" << endl;

  if (!CLI::HasParam("codes_file") && !CLI::HasParam("dictionary_file") &&
      !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --codes_file (-c), --dictionary_file (-d), nor "
        << "--output_model_file (-M) are specified; no output will be saved."
        << endl;

  if (CLI::HasParam("codes_file") && !CLI::HasParam("test_file"))
    Log::Fatal << "--codes_file (-c) is specified, but no test matrix ("
        << "specified with --test_file or -T) is given to encode!" << endl;

  if (!CLI::HasParam("training_file"))
  {
    if (CLI::HasParam("atoms"))
      Log::Warn << "--atoms (-k) ignored because --training_file (-t) is not "
          << "specified." << endl;
    if (CLI::HasParam("lambda"))
      Log::Warn << "--lambda (-l) ignored because --training_file (-t) is not "
          << "specified." << endl;
    if (CLI::HasParam("initial_dictionary"))
      Log::Warn << "--initial_dictionary (-i) ignored because --training_file "
          << "(-t) is not specified." << endl;
    if (CLI::HasParam("max_iterations"))
      Log::Warn << "--max_iterations (-n) ignored because --training_file (-t) "
          << "is not specified." << endl;
    if (CLI::HasParam("normalize"))
      Log::Warn << "--normalize (-N) ignored because --training_file (-t) is "
          << "not specified." << endl;
    if (CLI::HasParam("tolerance"))
      Log::Warn << "--tolerance (-o) ignored because --training_file (-t) is "
          << "not specified." << endl;
  }

  // Do we have an existing model?
  LocalCoordinateCoding lcc(0, 0.0);
  if (CLI::HasParam("input_model_file"))
  {
    data::Load(CLI::GetParam<string>("input_model_file"), "lcc_model", lcc,
        true);
  }

  if (CLI::HasParam("training_file"))
  {
    mat matX;
    data::Load(CLI::GetParam<string>("training_file"), matX, true);

    // Normalize each point if the user asked for it.
    if (CLI::HasParam("normalize"))
    {
      Log::Info << "Normalizing data before coding..." << endl;
      for (size_t i = 0; i < matX.n_cols; ++i)
        matX.col(i) /= norm(matX.col(i), 2);
    }

    lcc.Lambda() = CLI::GetParam<double>("lambda");
    lcc.Atoms() = (size_t) CLI::GetParam<int>("atoms");
    lcc.MaxIterations() = (size_t) CLI::GetParam<int>("max_iterations");
    lcc.Tolerance() = CLI::GetParam<double>("tolerance");

    // Inform the user if we are overwriting their model.
    if (CLI::HasParam("input_model_file"))
    {
      Log::Info << "Using dictionary from existing model in '"
          << CLI::GetParam<string>("input_model_file") << "' as initial "
          << "dictionary for training." << endl;
      lcc.Train<NothingInitializer>(matX);
    }
    else if (CLI::HasParam("initial_dictionary"))
    {
      // Load initial dictionary directly into LCC object.
      data::Load(CLI::GetParam<string>("initial_dictionary"), lcc.Dictionary(),
          true);

      // Validate the size of the initial dictionary.
      if (lcc.Dictionary().n_cols != lcc.Atoms())
      {
        Log::Fatal << "The initial dictionary has " << lcc.Dictionary().n_cols
            << " atoms, but the number of atoms was specified to be "
            << lcc.Atoms() << "!" << endl;
      }

      if (lcc.Dictionary().n_rows != matX.n_rows)
      {
        Log::Fatal << "The initial dictionary has " << lcc.Dictionary().n_rows
            << " dimensions, but the data has " << matX.n_rows << " dimensions!"
            << endl;
      }

      // Train the model.
      lcc.Train<NothingInitializer>(matX);
    }
    else
    {
      // Run with the default initialization.
      lcc.Train(matX);
    }
  }

  // Now, do we have any matrix to encode?
  if (CLI::HasParam("test_file"))
  {
    mat matY;
    data::Load(CLI::GetParam<string>("test_file"), matY, true);

    if (matY.n_rows != lcc.Dictionary().n_rows)
      Log::Fatal << "Model was trained with a dimensionality of "
          << lcc.Dictionary().n_rows << ", but data in test file "
          << CLI::GetParam<string>("test_file") << " has a dimensionality of "
          << matY.n_rows << "!" << endl;

    // Normalize each point if the user asked for it.
    if (CLI::HasParam("normalize"))
    {
      Log::Info << "Normalizing test data before coding..." << endl;
      for (size_t i = 0; i < matY.n_cols; ++i)
        matY.col(i) /= norm(matY.col(i), 2);
    }

    mat codes;
    lcc.Encode(matY, codes);

    if (CLI::HasParam("codes_file"))
      data::Save(CLI::GetParam<string>("codes_file"), codes);
  }

  // Did the user want to save the dictionary?
  if (CLI::HasParam("dictionary_file"))
    data::Save(CLI::GetParam<string>("dictionary_file"), lcc.Dictionary());

  // Did the user want to save the model?
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "lcc_model", lcc,
        false); // Non-fatal on failure.
}
