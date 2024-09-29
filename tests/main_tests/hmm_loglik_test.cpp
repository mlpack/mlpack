/**
 * @file tests/main_tests/hmm_loglik_test.cpp
 * @author Daivik Nema
 *
 * Test RUN_BINDING() of hmm_loglik_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/hmm_loglik_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

#include "hmm_test_utils.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(HMMLoglikTestFixture);

TEST_CASE_METHOD(HMMLoglikTestFixture, "HMMLoglikOutputNegativeTest",
                 "[HMMLoglikMainTest][BindingTests]")
{
  // Load data to train a discrete HMM model with.
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};

  // Initialize and train an HMM model.
  HMMModel* h = new HMMModel(DiscreteHMM);
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(params, &trainSeq);
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(params, &trainSeq);

  // Set the params for the hmm_loglik invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  RUN_BINDING();

  double loglik = params.Get<double>("log_likelihood");

  // Since the log of a probability <= 0 ...
  REQUIRE(loglik <= 0);
}
