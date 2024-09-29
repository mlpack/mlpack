/**
 * @file tests/kernel_pca_test.cpp
 * @author Ryan Curtin
 *
 * Test file for Kernel PCA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/kernel_pca.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;

/**
 * If KernelPCA is working right, then it should turn a circle dataset into a
 * linearly separable dataset in one dimension (which is easy to check).
 */
TEST_CASE("CircleTransformationTestNaive", "[KernelPCATest]")
{
  // The dataset, which will have three concentric rings in three dimensions.
  arma::mat dataset;

  // Now, there are 750 points centered at the origin with unit variance.
  dataset.randn(3, 750);
  dataset *= 0.05;

  // Take the second 250 points and spread them away from the origin.
  for (size_t i = 250; i < 500; ++i)
  {
    // Push the point away from the origin by 2.
    const double pointNorm = norm(dataset.col(i), 2);

    dataset(0, i) += 2.0 * (dataset(0, i) / pointNorm);
    dataset(1, i) += 2.0 * (dataset(1, i) / pointNorm);
    dataset(2, i) += 2.0 * (dataset(2, i) / pointNorm);
  }

  // Take the third 500 points and spread them away from the origin.
  for (size_t i = 500; i < 750; ++i)
  {
    // Push the point away from the origin by 5.
    const double pointNorm = norm(dataset.col(i), 2);

    dataset(0, i) += 5.0 * (dataset(0, i) / pointNorm);
    dataset(1, i) += 5.0 * (dataset(1, i) / pointNorm);
    dataset(2, i) += 5.0 * (dataset(2, i) / pointNorm);
  }

  // Now we have a dataset; we will use the GaussianKernel to perform KernelPCA
  // using the naive method to take it down to one dimension.
  KernelPCA<GaussianKernel> p;
  p.Apply(dataset, 1);

  // Get the ranges of each "class".  These are all initialized as empty ranges
  // containing no points.
  Range ranges[3];
  ranges[0] = Range();
  ranges[1] = Range();
  ranges[2] = Range();

  // Expand the ranges to hold all of the points in the class.
  for (size_t i = 0; i < 250; ++i)
    ranges[0] |= dataset(0, i);
  for (size_t i = 250; i < 500; ++i)
    ranges[1] |= dataset(0, i);
  for (size_t i = 500; i < 750; ++i)
    ranges[2] |= dataset(0, i);

  // None of these ranges should overlap -- the classes should be linearly
  // separable.
  REQUIRE(ranges[0].Contains(ranges[1]) == false);
  REQUIRE(ranges[0].Contains(ranges[2]) == false);
  REQUIRE(ranges[1].Contains(ranges[2]) == false);
}

/**
 * If KernelPCA is working right, then it should turn a circle dataset into a
 * linearly separable dataset in one dimension (which is easy to check).
 */
TEST_CASE("CircleTransformationTestNystroem", "[KernelPCATest]")
{
  // The dataset, which will have three concentric rings in three dimensions.
  arma::mat dataset;

  // Now, there are 750 points centered at the origin with unit variance.
  dataset.randn(3, 750);
  dataset *= 0.05;

  // Take the second 250 points and spread them away from the origin.
  for (size_t i = 250; i < 500; ++i)
  {
    // Push the point away from the origin by 2.
    const double pointNorm = norm(dataset.col(i), 2);

    dataset(0, i) += 2.0 * (dataset(0, i) / pointNorm);
    dataset(1, i) += 2.0 * (dataset(1, i) / pointNorm);
    dataset(2, i) += 2.0 * (dataset(2, i) / pointNorm);
  }

  // Take the third 500 points and spread them away from the origin.
  for (size_t i = 500; i < 750; ++i)
  {
    // Push the point away from the origin by 5.
    const double pointNorm = norm(dataset.col(i), 2);

    dataset(0, i) += 5.0 * (dataset(0, i) / pointNorm);
    dataset(1, i) += 5.0 * (dataset(1, i) / pointNorm);
    dataset(2, i) += 5.0 * (dataset(2, i) / pointNorm);
  }

  // Now we have a dataset; we will use the GaussianKernel to perform KernelPCA
  // using the nytroem method to take it down to one dimension.
  KernelPCA<GaussianKernel, NystroemKernelRule<GaussianKernel> > p;
  p.Apply(dataset, 1);

  // Get the ranges of each "class".  These are all initialized as empty ranges
  // containing no points.
  Range ranges[3];
  ranges[0] = Range();
  ranges[1] = Range();
  ranges[2] = Range();

  // Expand the ranges to hold all of the points in the class.
  for (size_t i = 0; i < 250; ++i)
    ranges[0] |= dataset(0, i);
  for (size_t i = 250; i < 500; ++i)
    ranges[1] |= dataset(0, i);
  for (size_t i = 500; i < 750; ++i)
    ranges[2] |= dataset(0, i);

  // None of these ranges should overlap -- the classes should be linearly
  // separable.
  REQUIRE(ranges[0].Contains(ranges[1]) == false);
  REQUIRE(ranges[0].Contains(ranges[2]) == false);
  REQUIRE(ranges[1].Contains(ranges[2]) == false);
}
