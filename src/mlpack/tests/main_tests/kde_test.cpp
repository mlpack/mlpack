/**
 * @file kde_test.cpp
 * @author Roberto Hueso
 *
 * Test mlpackMain() of kde_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "KDE";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/kde/kde_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KDETestFixture
{
 public:
  KDETestFixture()
  {
      // Cache in the options for this program.
      CLI::RestoreSettings(testName);
  }

  ~KDETestFixture()
  {
      // Clear the settings.
      CLI::ClearSettings();
  }
};

void ResetKDESettings()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(KDEMainTest, KDETestFixture);

/**
  * Ensure that the estimations we get for KDEMain, are the same as the ones we
  * get from the KDE class without any wrappers.
 **/
BOOST_AUTO_TEST_CASE(KDEEqualResultsForMain)
{
  // Datasets
  arma::mat reference = arma::randu(3, 500);
  arma::mat query = arma::randu(3, 100);
  arma::vec kdeEstimations, mainEstimations;
  double kernelBandwidth = 1.5;
  double relError = 0.05;

  kernel::GaussianKernel kernel(kernelBandwidth);
  metric::EuclideanDistance metric;
  KDE<metric::EuclideanDistance,
      arma::mat,
      kernel::GaussianKernel,
      tree::RTree>
    kde(metric, kernel, relError, 0.0);
  kde.Train(reference);
  kde.Evaluate(query, kdeEstimations);
  // Normalize estimations
  kdeEstimations /= kernel.Normalizer(reference.n_rows);

  // Main estimations
  SetInputParam("reference", reference);
  SetInputParam("query", query);
  SetInputParam("kernel", std::string("gaussian"));
  SetInputParam("tree", std::string("r-tree"));
  SetInputParam("rel_error", relError);
  SetInputParam("bandwidth", kernelBandwidth);

  mlpackMain();

  mainEstimations = std::move(CLI::GetParam<arma::mat>("output"));

  // Check whether results are equal.
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(kdeEstimations[i], mainEstimations[i], relError);
}

/**
  * Ensuring that absence of input data is checked.
 **/
BOOST_AUTO_TEST_CASE(KDENoInputData)
{
  // No input data is not provided. Should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
