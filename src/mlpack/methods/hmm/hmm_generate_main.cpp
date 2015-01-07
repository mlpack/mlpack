/**
 * @file hmm_viterbi_main.cpp
 * @author Ryan Curtin
 *
 * Compute the most probably hidden state sequence of a given observation
 * sequence for a given HMM.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
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

PARAM_STRING_REQ("model_file", "File containing HMM (XML).", "m");
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

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Set random seed.
  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Load observations.
  const string modelFile = CLI::GetParam<string>("model_file");
  const int length = CLI::GetParam<int>("length");
  const int startState = CLI::GetParam<int>("start_state");

  if (length <= 0)
  {
    Log::Fatal << "Invalid sequence length (" << length << "); must be greater "
        << "than or equal to 0!" << endl;
  }

  // Load model, but first we have to determine its type.
  SaveRestoreUtility sr;
  sr.ReadFile(modelFile);
  string type;
  sr.LoadParameter(type, "hmm_type");

  mat observations;
  Col<size_t> sequence;
  if (type == "discrete")
  {
    HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));

    LoadHMM(hmm, sr);

    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
      Log::Fatal << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!" << endl;
    }

    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
  }
  else if (type == "gaussian")
  {
    HMM<GaussianDistribution> hmm(1, GaussianDistribution(1));

    LoadHMM(hmm, sr);

    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
      Log::Fatal << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!" << endl;
    }

    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
  }
  else if (type == "gmm")
  {
    HMM<GMM<> > hmm(1, GMM<>(1, 1));

    LoadHMM(hmm, sr);

    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
      Log::Fatal << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!" << endl;
    }

    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
  }
  else
  {
    Log::Fatal << "Unknown HMM type '" << type << "' in file '" << modelFile
        << "'!" << endl;
  }

  // Save observations.
  const string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, observations, true);

  // Do we want to save the hidden sequence?
  const string sequenceFile = CLI::GetParam<string>("state_file");
  if (sequenceFile != "")
    data::Save(sequenceFile, sequence, true);
}
