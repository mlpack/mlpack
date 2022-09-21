/**
 * @file methods/gmm/gmm_generate_main.cpp
 * @author Ryan Curtin
 *
 * Load a GMM from file, then generate samples from it.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME gmm_generate

#include <mlpack/core/util/mlpack_main.hpp>
#include "gmm.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("GMM Sample Generator");

// Short description.
BINDING_SHORT_DESC(
    "A sample generator for pre-trained GMMs.  Given a pre-trained GMM, this "
    "can sample new points randomly from that distribution.");

// Long description.
BINDING_LONG_DESC(
    "This program is able to generate samples from a pre-trained GMM (use "
    "gmm_train to train a GMM).  The pre-trained GMM must be specified with "
    "the " + PRINT_PARAM_STRING("input_model") + " parameter.  The number "
    "of samples to generate is specified by the " +
    PRINT_PARAM_STRING("samples") + " parameter.  Output samples may be "
    "saved with the " + PRINT_PARAM_STRING("output") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "The following command can be used to generate 100 samples from the pre-"
    "trained GMM " + PRINT_MODEL("gmm") + " and store those generated "
    "samples in " + PRINT_DATASET("samples") + ":"
    "\n\n" +
    PRINT_CALL("gmm_generate", "input_model", "gmm", "samples", 100, "output",
        "samples"));

// See also...
BINDING_SEE_ALSO("@gmm_train", "#gmm_train");
BINDING_SEE_ALSO("@gmm_probability", "#gmm_probability");
BINDING_SEE_ALSO("Gaussian Mixture Models on Wikipedia",
    "https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model");
BINDING_SEE_ALSO("GMM class documentation", "@src/mlpack/methods/gmm/gmm.hpp");

PARAM_MODEL_IN_REQ(GMM, "input_model", "Input GMM model to generate samples "
    "from.", "m");
PARAM_INT_IN_REQ("samples", "Number of samples to generate.", "n");

PARAM_MATRIX_OUT("output", "Matrix to save output samples in.", "o");

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Parameter sanity checks.
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no results will be saved");

  if (params.Get<int>("seed") == 0)
    RandomSeed(time(NULL));
  else
    RandomSeed((size_t) params.Get<int>("seed"));

  RequireParamValue<int>(params, "samples", [](int x) { return x > 0; }, true,
      "number of samples must be greater than 0");

  GMM* gmm = params.Get<GMM*>("input_model");

  size_t length = (size_t) params.Get<int>("samples");
  Log::Info << "Generating " << length << " samples..." << endl;
  arma::mat samples(gmm->Dimensionality(), length);
  for (size_t i = 0; i < length; ++i)
    samples.col(i) = gmm->Random();

  // Save, if the user asked for it.
  params.Get<arma::mat>("output") = std::move(samples);
}
