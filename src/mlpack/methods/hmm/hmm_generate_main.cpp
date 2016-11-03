/**
 * @file hmm_generate_main.cpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Compute the most probably hidden state sequence of a given observation
 * sequence for a given HMM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

PARAM_STRING_IN_REQ("model_file", "File containing HMM.", "m");
PARAM_INT_IN_REQ("length", "Length of sequence to generate.", "l");

PARAM_INT_IN("start_state", "Starting state of sequence.", "t", 0);
PARAM_MATRIX_OUT("output", "Matrix to save observation sequence to.", "o");
PARAM_UMATRIX_OUT("state", "Matrix to save hidden state sequence to.", "S");
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

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
    if (CLI::HasParam("output"))
      CLI::GetParam<mat>("output") = std::move(observations);

    // Do we want to save the hidden sequence?
    if (CLI::HasParam("state"))
      CLI::GetParam<Mat<size_t>>("state") = std::move(sequence);
  }
};

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("output") && !CLI::HasParam("state"))
    Log::Warn << "Neither --output_file nor --state_file are specified; no "
        << "output will be saved!" << endl;

  // Set random seed.
  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Load model, and perform the generation.
  const string modelFile = CLI::GetParam<string>("model_file");
  LoadHMMAndPerformAction<Generate>(modelFile);
}
