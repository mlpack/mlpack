/**
 * @file tests/kernel_svm_test.cpp
 * @author Himanshu Pathak
 *
 * Test the Kernel SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/kernel_svm/kernel_svm.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(KernelSVMTest);

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using smo optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFitIntercept)
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;
  const double delta = 1.0;

  // Generate a two-Gaussian dataset,
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat data(inputSize, points);
    arma::Row<size_t> labels(points);
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels[i] = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels[i] = 1;
    }

    // Now train a svm object on it.
    KernelSVM<> svm(data, labels, 2, 1.0, 10.0, false, 10);

    // Ensure that the error is close to zero.
    const double acc = svm.ComputeAccuracy(data, labels);
    if (acc <= 0.98)
      continue;

    arma::mat testData(inputSize, points);
    arma::Row<size_t> testLabels(inputSize);

    // Create a test set.
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels[i] = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels[i] = 1;
    }

    // Ensure that the error is close to zero.
    const double testAcc = svm.ComputeAccuracy(data, labels);
    if (testAcc >= 0.95)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

BOOST_AUTO_TEST_SUITE_END();