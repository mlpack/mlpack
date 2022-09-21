/**
 * @file methods/hmm/hmm_loglik_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME hmm_loglik

#include <mlpack/core/util/mlpack_main.hpp>

#include "hmm.hpp"
#include "hmm_model.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Hidden Markov Model (HMM) Sequence Log-Likelihood");

// Short description.
BINDING_SHORT_DESC(
    "A utility for computing the log-likelihood of a sequence for Hidden Markov"
    " Models (HMMs).  Given a pre-trained HMM and an observation sequence, this"
    " computes and returns the log-likelihood of that sequence being observed "
    "from that HMM.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes an already-trained HMM, specified with the " +
    PRINT_PARAM_STRING("input_model") + " parameter, and evaluates the "
    "log-likelihood of a sequence of observations, given with the " +
    PRINT_PARAM_STRING("input") + " parameter.  The computed log-likelihood is"
    " given as output.");

// Example.
BINDING_EXAMPLE(
    "For example, to compute the log-likelihood of the sequence " +
    PRINT_DATASET("seq") + " with the pre-trained HMM " + PRINT_MODEL("hmm") +
    ", the following command may be used: "
    "\n\n" +
    PRINT_CALL("hmm_loglik", "input", "seq", "input_model", "hmm"));

// See also...
BINDING_SEE_ALSO("@hmm_train", "#hmm_train");
BINDING_SEE_ALSO("@hmm_generate", "#hmm_generate");
BINDING_SEE_ALSO("@hmm_viterbi", "#hmm_viterbi");
BINDING_SEE_ALSO("Hidden Mixture Models on Wikipedia",
    "https://en.wikipedia.org/wiki/Hidden_Markov_model");
BINDING_SEE_ALSO("HMM class documentation", "@src/mlpack/methods/hmm/hmm.hpp");

PARAM_MATRIX_IN_REQ("input", "File containing observations,", "i");
PARAM_MODEL_IN_REQ(HMMModel, "input_model", "File containing HMM.", "m");

PARAM_DOUBLE_OUT("log_likelihood", "Log-likelihood of the sequence.");

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Loglik
{
  template<typename HMMType>
  static void Apply(util::Params& params, HMMType& hmm, void* /* extraInfo */)
  {
    // Load the data sequence.
    mat dataSeq = std::move(params.Get<mat>("input"));

    // Detect if we need to transpose the data, in the case where the input data
    // has one dimension.
    if ((dataSeq.n_cols == 1) && (hmm.Emission()[0].Dimensionality() == 1))
    {
      Log::Info << "Data sequence appears to be transposed; correcting."
          << endl;
      dataSeq = dataSeq.t();
    }

    if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
    {
      Log::Fatal << "Dimensionality of sequence (" << dataSeq.n_rows << ") is "
          << "not equal to the dimensionality of the HMM ("
          << hmm.Emission()[0].Dimensionality() << ")!" << endl;
    }

    const double loglik = hmm.LogLikelihood(dataSeq);

    params.Get<double>("log_likelihood") = loglik;
  }
};

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Load model, and calculate the log-likelihood of the sequence.
  params.Get<HMMModel*>("input_model")->PerformAction<Loglik>(
      params, (void*) NULL);
}
