/**
 * @file tests/main_tests/adaboost_fit_test.cpp
 * @author Nippun Sharma
 *
 * Test RUN_BINDING() of adaboost_fit_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methoods/adaboost/adaboost_fit_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AdaBoostFitMainTestFixture);

// Test if the error is thrown for invalid tolerance.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitToleranceTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("tolerance", -1); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Test if error is thrown for invalid iterations.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitIterationsTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("iterations", -1); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Test if error is thrown for invalid weak learner.
TEST_CASE_METHOD(AdaBoostFitMainTestFixture, "AdaBoostFitWeakLearnerTest",
                 "[AdaBoostFitMainTest][BindingTests]")
{
  arma::mat trainData;
  if (!data::Load("vc2.csv", trainData))
    FAIL("Unable to load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    FAIL("Unable to load label dataset vc2_labels.txt!");

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weak_learner", "xyz"); // Invalid!

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROW_AS(RUN_BINDING(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
