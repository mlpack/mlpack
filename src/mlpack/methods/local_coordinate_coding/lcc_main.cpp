/**
 * @file lcc_main.cpp
 * @author Nishant Mehta
 *
 * Executable for Local Coordinate Coding.
 *
 * This file is part of MLPACK 1.0.6.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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

PARAM_STRING_REQ("input_file", "Filename of the input data.", "i");
PARAM_INT_REQ("atoms", "Number of atoms in the dictionary.", "k");

PARAM_DOUBLE("lambda", "Weighted l1-norm regularization parameter.", "l", 0.0);

PARAM_INT("max_iterations", "Maximum number of iterations for LCC (0 indicates "
    "no limit).", "n", 0);

PARAM_STRING("initial_dictionary", "Filename for optional initial dictionary.",
    "D", "");

PARAM_STRING("dictionary_file", "Filename to save the output dictionary to.",
    "d", "dictionary.csv");
PARAM_STRING("codes_file", "Filename to save the output codes to.", "c",
    "codes.csv");

PARAM_FLAG("normalize", "If set, the input data matrix will be normalized "
    "before coding.", "N");

PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

PARAM_DOUBLE("objective_tolerance", "Tolerance for objective function.", "o",
    0.01);

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

  const double lambda = CLI::GetParam<double>("lambda");

  const string inputFile = CLI::GetParam<string>("input_file");
  const string dictionaryFile = CLI::GetParam<string>("dictionary_file");
  const string codesFile = CLI::GetParam<string>("codes_file");
  const string initialDictionaryFile =
      CLI::GetParam<string>("initial_dictionary");

  const size_t maxIterations = CLI::GetParam<int>("max_iterations");
  const size_t atoms = CLI::GetParam<int>("atoms");

  const bool normalize = CLI::HasParam("normalize");

  const double objTolerance = CLI::GetParam<double>("objective_tolerance");

  mat input;
  data::Load(inputFile, input, true);

  Log::Info << "Loaded " << input.n_cols << " point in " << input.n_rows
      << " dimensions." << endl;

  // Normalize each point if the user asked for it.
  if (normalize)
  {
    Log::Info << "Normalizing data before coding..." << endl;
    for (size_t i = 0; i < input.n_cols; ++i)
      input.col(i) /= norm(input.col(i), 2);
  }

  // If there is an initial dictionary, be sure we do not initialize one.
  if (initialDictionaryFile != "")
  {
    LocalCoordinateCoding<NothingInitializer> lcc(input, atoms, lambda);

    // Load initial dictionary directly into LCC object.
    data::Load(initialDictionaryFile, lcc.Dictionary(), true);

    // Validate size of initial dictionary.
    if (lcc.Dictionary().n_cols != atoms)
    {
      Log::Fatal << "The initial dictionary has " << lcc.Dictionary().n_cols
          << " atoms, but the number of atoms was specified to be " << atoms
          << "!" << endl;
    }

    if (lcc.Dictionary().n_rows != input.n_rows)
    {
      Log::Fatal << "The initial dictionary has " << lcc.Dictionary().n_rows
          << " dimensions, but the data has " << input.n_rows << " dimensions!"
          << endl;
    }

    // Run LCC.
    lcc.Encode(maxIterations, objTolerance);

    // Save the results.
    Log::Info << "Saving dictionary matrix to '" << dictionaryFile << "'.\n";
    data::Save(dictionaryFile, lcc.Dictionary());
    Log::Info << "Saving sparse codes to '" << codesFile << "'.\n";
    data::Save(codesFile, lcc.Codes());
  }
  else
  {
    // No initial dictionary.
    LocalCoordinateCoding<> lcc(input, atoms, lambda);

    // Run LCC.
    lcc.Encode(maxIterations, objTolerance);

    // Save the results.
    Log::Info << "Saving dictionary matrix to '" << dictionaryFile << "'.\n";
    data::Save(dictionaryFile, lcc.Dictionary());
    Log::Info << "Saving sparse codes to '" << codesFile << "'.\n";
    data::Save(codesFile, lcc.Codes());
  }
}
