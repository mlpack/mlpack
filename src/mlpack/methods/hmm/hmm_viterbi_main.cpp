/**
 * @file hmm_viterbi_main.cpp
 * @author Ryan Curtin
 *
 * Compute the most probably hidden state sequence of a given observation
 * sequence for a given HMM.
 *
 * This file is part of mlpack 2.0.2.
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

PROGRAM_INFO("Hidden Markov Model (HMM) Viterbi State Prediction", "This "
    "utility takes an already-trained HMM (--model_file) and evaluates the "
    "most probably hidden state sequence of a given sequence of observations "
    "(--input_file), using the Viterbi algorithm.  The computed state sequence "
    "is saved to the specified output file (--output_file).");

PARAM_STRING_REQ("input_file", "File containing observations,", "i");
PARAM_STRING_REQ("model_file", "File containing HMM.", "m");
PARAM_STRING("output_file", "File to save predicted state sequence to.",
    "o", "");

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace arma;
using namespace std;

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Viterbi
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, void* /* extraInfo */)
  {
    // Load observations.
    const string inputFile = CLI::GetParam<string>("input_file");
    const string outputFile = CLI::GetParam<string>("output_file");

    mat dataSeq;
    data::Load(inputFile, dataSeq, true);

    // See if transposing the data could make it the right dimensionality.
    if ((dataSeq.n_cols == 1) && (hmm.Emission()[0].Dimensionality() == 1))
    {
      Log::Info << "Data sequence appears to be transposed; correcting."
          << endl;
      dataSeq = dataSeq.t();
    }

    // Verify correct dimensionality.
    if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
      Log::Fatal << "Observation dimensionality (" << dataSeq.n_rows << ") "
          << "does not match HMM Gaussian dimensionality ("
          << hmm.Emission()[0].Dimensionality() << ")!" << endl;

    arma::Row<size_t> sequence;
    hmm.Predict(dataSeq, sequence);

    // Save output.
    if (CLI::HasParam("output_file"))
      data::Save(outputFile, sequence, true);
  }
};

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("output_file"))
    Log::Warn << "--output_file (-o) is not specified; no results will be "
        << "saved!" << endl;

  const string modelFile = CLI::GetParam<string>("model_file");
  LoadHMMAndPerformAction<Viterbi>(modelFile);
}
