/**
 * @file hmm_generate_main.cpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Compute the most probably hidden state sequence of a given observation
 * sequence for a given HMM.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include "hmm.hpp"
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

PROGRAM_INFO("Hidden Markov Model (HMM) Sequence Generator", "This "
    "utility takes an already-trained HMM (--model_file) and generates a "
    "random observation sequence and hidden state sequence based on its "
    "parameters, saving them to the specified files (--output_file and "
    "--state_file)");

PARAM_STRING_REQ("model_file", "File containing HMM.", "m");
PARAM_INT_REQ("length", "Length of sequence to generate.", "l");

PARAM_INT("start_state", "Starting state of sequence.", "t", 0);
PARAM_STRING("output_file", "File to save observation sequence to.", "o",
    "output.csv");
PARAM_STRING("state_file", "File to save hidden state sequence to (may be left "
    "unspecified.", "S", "");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

// Because we don't know what the type of our HMM is, we need to write a
// function which can take arbitrary HMM types.
struct Generate
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, void* /* extraInfo */)
  {
    mat observations;
    Row<size_t> sequence;

    // Load the parameters.
    const size_t startState = (size_t) CLI::GetParam<int>("start_state");
    const size_t length = (size_t) CLI::GetParam<int>("length");

    Log::Info << "Generating sequence of length " << length << "..." << endl;
    if (startState >= hmm.Transition().n_rows)
      Log::Fatal << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!" << endl;

    hmm.Generate(length, observations, sequence, startState);

    // Now save the output.
    const string outputFile = CLI::GetParam<string>("output_file");
    data::Save(outputFile, observations, true);

    // Do we want to save the hidden sequence?
    const string sequenceFile = CLI::GetParam<string>("state_file");
    if (sequenceFile != "")
      data::Save(sequenceFile, sequence, true);
  }
};

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Set random seed.
  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Load model, and perform the generation.
  const string modelFile = CLI::GetParam<string>("model_file");
  LoadHMMAndPerformAction<Generate>(modelFile);
}
