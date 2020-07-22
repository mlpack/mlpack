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
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(KernelSVMTest);

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using smo optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFitIntercept)
{
  const size_t points = 1000;
  const size_t inputSize = 3;

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
    KernelSVM<> svm(data, labels, 1.0, false, 10);

    // Ensure that the error is close to zero.
    const double acc = svm.ComputeAccuracy(data, labels);
    if (acc >= 0.92)
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

/**
 * Test training of kernel svm for two classes on a mnist 4
 * and 9 dataset.
 */
BOOST_AUTO_TEST_CASE(GaussianKernelSVMMnistDataset)
{
  arma::mat dataset;
  dataset.load("mnist_first250_training_4s_and_9s.arm");

  // Normalize each point since these are images.
  for (size_t i = 0; i < dataset.n_cols; ++i)
    dataset.col(i) /= norm(dataset.col(i), 2);

  arma::Row<size_t> labels(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols / 2; ++i)
  {
    labels[i] = 0;
  }

  for (size_t i = dataset.n_cols / 2; i < dataset.n_cols; ++i)
  {
    labels[i] = 1;
  }

  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    // Now train a svm object on it.
    KernelSVM<arma::mat,
              kernel::GaussianKernel> svm(dataset,
                                labels, 1.0, true, 10);
    // Ensure that the error is close to zero.
    const double testAcc = svm.ComputeAccuracy(dataset, labels);
    if (testAcc >= 0.95)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

/**
 * Test training of kernel svm on concentric circle dataset.
 */
BOOST_AUTO_TEST_CASE(ConcentricCircleDataset)
{
  // The dataset, which will have three concentric rings in three dimensions.
  arma::mat dataset;

  // Now, there are 500 points centered at the origin with unit variance.
  dataset = arma::randn(3, 500);
  dataset *= 0.05;
  arma::Row<size_t> labels(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols / 2; ++i)
  {
    labels[i] = 0;
  }


  // Take the second 250 points and spread them away from the origin.
  for (size_t i = 250; i < 500; ++i)
  {
    // Push the point away from the origin by 2.
    const double pointNorm = norm(dataset.col(i), 2);

    dataset(0, i) += 2.0 * (dataset(0, i) / pointNorm);
    dataset(1, i) += 2.0 * (dataset(1, i) / pointNorm);
    dataset(2, i) += 2.0 * (dataset(2, i) / pointNorm);
    labels[i] = 1;
  }

  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    // Now train a svm object on it.
    KernelSVM<arma::mat,
              kernel::GaussianKernel> svm(dataset,
                                labels, 1.0, true, 10);
    std::cout<<"training complete"<<std::endl;
    // Ensure that the error is close to zero.
    const double testAcc = svm.ComputeAccuracy(dataset, labels);
    if (testAcc >= 0.95)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

BOOST_AUTO_TEST_SUITE_END();
