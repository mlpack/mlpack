/**
 * @file hmm_loglik_main.cpp
 * @author Ryan Curtin
 *
 * Compute the log-likelihood of a given sequence for a given HMM.
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

PROGRAM_INFO("Hidden Markov Model (HMM) Sequence Log-Likelihood", "This "
    "utility takes an already-trained HMM (--model_file) and evaluates the "
    "log-likelihood of a given sequence of observations (--input_file).  The "
    "computed log-likelihood is given directly to stdout.");

PARAM_MATRIX_IN_REQ("input", "File containing observations,", "i");
PARAM_STRING_IN_REQ("model_file", "File containing HMM.", "m");

PARAM_DOUBLE_OUT("log_likelihood", "Log-likelihood of the sequence.");

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace arma;
using namespace std;

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Loglik
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, void* /* extraInfo */)
  {
    // Load the data sequence.
    mat dataSeq = std::move(CLI::GetParam<mat>("input"));

    // Detect if we need to transpose the data, in the case where the input data
    // has one dimension.
    if ((dataSeq.n_cols == 1) && (hmm.Emission()[0].Dimensionality() == 1))
    {
      Log::Info << "Data sequence appears to be transposed; correcting."
          << endl;
      dataSeq = dataSeq.t();
    }

    if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
      Log::Fatal << "Dimensionality of sequence (" << dataSeq.n_rows << ") is "
          << "not equal to the dimensionality of the HMM ("
          << hmm.Emission()[0].Dimensionality() << ")!" << endl;

    const double loglik = hmm.LogLikelihood(dataSeq);

    CLI::GetParam<double>("log_likelihood") = loglik;
  }
};

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Load model, and calculate the log-likelihood of the sequence.
  const string modelFile = CLI::GetParam<string>("model_file");
  LoadHMMAndPerformAction<Loglik>(modelFile);
}
