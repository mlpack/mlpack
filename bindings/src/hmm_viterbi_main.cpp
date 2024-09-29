/**
 * @file methods/hmm/hmm_viterbi_main.cpp
 * @author Ryan Curtin
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

#undef BINDING_NAME
#define BINDING_NAME hmm_viterbi

#include <mlpack/core/util/mlpack_main.hpp>

#include "hmm.hpp"
#include "hmm_model.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Hidden Markov Model (HMM) Viterbi State Prediction");

// Short description.
BINDING_SHORT_DESC(
    "A utility for computing the most probable hidden state sequence for Hidden"
    " Markov Models (HMMs).  Given a pre-trained HMM and an observed sequence, "
    "this uses the Viterbi algorithm to compute and return the most probable "
    "hidden state sequence.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes an already-trained HMM, specified as " +
    PRINT_PARAM_STRING("input_model") + ", and evaluates the most probable "
    "hidden state sequence of a given sequence of observations (specified as "
    "'" + PRINT_PARAM_STRING("input") + ", using the Viterbi algorithm.  The "
    "computed state sequence may be saved using the " +
    PRINT_PARAM_STRING("output") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to predict the state sequence of the observations " +
    PRINT_DATASET("obs") + " using the HMM " + PRINT_MODEL("hmm") + ", "
    "storing the predicted state sequence to " + PRINT_DATASET("states") +
    ", the following command could be used:"
    "\n\n" +
    PRINT_CALL("hmm_viterbi", "input", "obs", "input_model", "hmm", "output",
        "states"));

// See also...
BINDING_SEE_ALSO("@hmm_train", "#hmm_train");
BINDING_SEE_ALSO("@hmm_generate", "#hmm_generate");
BINDING_SEE_ALSO("@hmm_loglik", "#hmm_loglik");
BINDING_SEE_ALSO("Hidden Mixture Models on Wikipedia",
    "https://en.wikipedia.org/wiki/Hidden_Markov_model");
BINDING_SEE_ALSO("HMM class documentation", "@src/mlpack/methods/hmm/hmm.hpp");

PARAM_MATRIX_IN_REQ("input", "Matrix containing observations,", "i");
PARAM_MODEL_IN_REQ(HMMModel, "input_model", "Trained HMM to use.", "m");
PARAM_UMATRIX_OUT("output", "File to save predicted state sequence to.", "o");

// Because we don't know what the type of our HMM is, we need to write a
// function that can take arbitrary HMM types.
struct Viterbi
{
  template<typename HMMType>
  static void Apply(util::Params& params, HMMType& hmm, void* /* extraInfo */)
  {
    // Load observations.
    mat dataSeq = std::move(params.Get<arma::mat>("input"));

    // See if transposing the data could make it the right dimensionality.
    if ((dataSeq.n_cols == 1) && (hmm.Emission()[0].Dimensionality() == 1))
    {
      Log::Info << "Data sequence appears to be transposed; correcting."
          << endl;
      dataSeq = dataSeq.t();
    }

    // Verify correct dimensionality.
    if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
    {
      Log::Fatal << "Observation dimensionality (" << dataSeq.n_rows << ") "
          << "does not match HMM Gaussian dimensionality ("
          << hmm.Emission()[0].Dimensionality() << ")!" << endl;
    }

    arma::Row<size_t> sequence;
    hmm.Predict(dataSeq, sequence);

    // Save output.
    params.Get<arma::Mat<size_t>>("output") = std::move(sequence);
  }
};

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no results will be saved");

  params.Get<HMMModel*>("input_model")->PerformAction<Viterbi>(
      params, (void*) NULL);
}
