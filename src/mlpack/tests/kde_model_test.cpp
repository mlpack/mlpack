/**
 * @file tests/kde_model_test.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// The test as of now only ensures that the header is self-contained.
// As soon as more useful tests are added, it should still be ensured
// that the main header is included first to preserve the self-containedness.
#include <mlpack/methods/kde/kde_model.hpp>

/**
 * @file tests/kde_model_test.cpp
 *
 * Tests for Kernel Density Estimation (KDE) model.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde_model.hpp>
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::kde;

/**
 * Test for basic construction and checking model parameters.
 */
TEST_CASE("KDEModelBasicTest", "[KDETest]")
{
  // Create a small dataset (2D).
  arma::mat dataset = { { 1.0, 2.0, 3.0 },
                        { 1.0, 2.0, 3.0 } };

  // Construct KDE model with default kernel (Gaussian).
  KDE<EuclideanDistance> kdeModel;

  // Fit the model.
  kdeModel.Train(dataset);

  // Check if the model has the expected number of components.
  REQUIRE(kdeModel.NumPoints() == 3);

  // Ensure that the model contains kernels after fitting.
  REQUIRE(kdeModel.KernelType() == KDE<EuclideanDistance>::GAUSSIAN);  // Should be Gaussian by default.
}

/**
 * Test for fitting the KDE model on a dataset and evaluating the density.
 */
TEST_CASE("KDEModelDensityTest", "[KDETest]")
{
  // Create a dataset of 2D points.
  arma::mat dataset = { { 1.0, 2.0, 3.0 },
                        { 1.0, 2.0, 3.0 } };

  // Construct and train the KDE model.
  KDE<EuclideanDistance> kdeModel;
  kdeModel.Train(dataset);

  // Test density for points near the data points.
  arma::vec queryPoint = {1.0, 1.0};
  double density = kdeModel.Evaluate(queryPoint);
  
  // Density at query point should not be zero (we trained on similar data).
  REQUIRE(density > 0);

  // Test density for points far from the data points (should be lower).
  queryPoint = {10.0, 10.0};
  density = kdeModel.Evaluate(queryPoint);
  
  // Density at query point far from the dataset should be very low or near zero.
  REQUIRE(density < 1e-5);
}

/**
 * Test for using different kernels in KDE.
 */
TEST_CASE("KDEModelKernelTest", "[KDETest]")
{
  // Create a simple dataset.
  arma::mat dataset = { { 1.0, 2.0, 3.0 },
                        { 1.0, 2.0, 3.0 } };

  // Test Gaussian kernel.
  KDE<EuclideanDistance> kdeGaussian;
  kdeGaussian.Train(dataset);
  REQUIRE(kdeGaussian.KernelType() == KDE<EuclideanDistance>::GAUSSIAN);

  // Test Epanechnikov kernel.
  KDE<EuclideanDistance> kdeEpanechnikov(KDE<EuclideanDistance>::EPANECHNIKOV);
  kdeEpanechnikov.Train(dataset);
  REQUIRE(kdeEpanechnikov.KernelType() == KDE<EuclideanDistance>::EPANECHNIKOV);
}

/**
 * Test for edge case: very small dataset.
 */
TEST_CASE("KDEModelSmallDatasetTest", "[KDETest]")
{
  // Create a small dataset with only one point.
  arma::mat dataset = { { 1.0 } };

  // Construct and train the KDE model.
  KDE<EuclideanDistance> kdeModel;
  kdeModel.Train(dataset);

  // Ensure that the density is non-zero for the trained point.
  arma::vec queryPoint = { 1.0 };
  double density = kdeModel.Evaluate(queryPoint);
  REQUIRE(density > 0);

  // Ensure that the density is zero for a point far away from the dataset.
  queryPoint = { 10.0 };
  density = kdeModel.Evaluate(queryPoint);
  REQUIRE(density == Approx(0.0).epsilon(1e-5));
}

/**
 * Test for invalid input: empty dataset.
 */
TEST_CASE("KDEModelEmptyDatasetTest", "[KDETest]")
{
  // Create an empty dataset.
  arma::mat dataset;

  // Construct KDE model.
  KDE<EuclideanDistance> kdeModel;

  // Ensure that training on an empty dataset throws an exception.
  REQUIRE_THROWS_AS(kdeModel.Train(dataset), std::invalid_argument);
}

/**
 * Test for saving and loading the KDE model.
 */
TEST_CASE("KDEModelSerializationTest", "[KDETest]")
{
  // Create a simple dataset.
  arma::mat dataset = { { 1.0, 2.0, 3.0 },
                        { 1.0, 2.0, 3.0 } };

  // Train the KDE model.
  KDE<EuclideanDistance> kdeModel;
  kdeModel.Train(dataset);

  // Save the model.
  mlpack::util::Save("kde_model.xml", "model", kdeModel);

  // Load the model back.
  KDE<EuclideanDistance> loadedModel;
  mlpack::util::Load("kde_model.xml", "model", loadedModel);

  // Ensure that the loaded model is equivalent to the original.
  REQUIRE(kdeModel.NumPoints() == loadedModel.NumPoints());
  REQUIRE(kdeModel.KernelType() == loadedModel.KernelType());

  // Clean up saved file.
  std::remove("kde_model.xml");
}
