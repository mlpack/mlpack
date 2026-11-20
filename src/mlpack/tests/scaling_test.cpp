/**
 * @file tests/scaling_test.cpp
 *
 * Tests for Scaling of dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace std;

/**
 * Test For MinMax Scaler Class.
 */
TEMPLATE_TEST_CASE("MinMaxScalerTest", "[ScalingTest][tiny]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
  MatType scaled = "0 0.2500 0.5000 1.000;"
                   "0 0.2500 0.5000 1.000;";
  MatType scaleddataset, temp;
  MinMaxScaler<MatType> scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled, tolerance);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test For MaxAbs Scaler Class.
 */
TEMPLATE_TEST_CASE("MaxAbsScalerTest", "[ScalingTest][tiny]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
  MatType scaled = "-1 -0.5 0 1;"
                   "0.1111111111 0.3333333333 0.55555556 1.0000;";
  MatType scaleddataset, temp;
  MaxAbsScaler<MatType> scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled, tolerance);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test For Standard Scaler Class.
 */
TEMPLATE_TEST_CASE("StandardScalerTest", "[ScalingTest][tiny]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
  MatType scaled = "-1.18321596 -0.50709255  0.16903085 1.52127766;"
                   "-1.18321596 -0.50709255  0.16903085 1.52127766;";
  MatType scaleddataset, temp;
  StandardScaler<MatType> scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled, tolerance);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test For MeanNormalization Scaler Class.
 */
TEMPLATE_TEST_CASE("MeanNormalizationTest", "[ScalingTest][tiny]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
  MatType scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                   "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
  MatType scaleddataset, temp;
  MeanNormalization<MatType> scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled, tolerance);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test to pass same matrix as input and output
 */
TEMPLATE_TEST_CASE("SameInputOutputTest", "[ScalingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
  MatType temp = dataset;
  MatType scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                   "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
  MeanNormalization<MatType> scale;
  scale.Fit(temp);
  scale.Transform(temp, temp);
  CheckMatrices(scaled, temp, tolerance);
  scale.InverseTransform(temp, temp);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test for Zero Matrix.
 */
TEMPLATE_TEST_CASE("ZeroMatrixTest", "[ScalingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType input(2, 4);
  MatType temp;
  MeanNormalization<MatType> scale;
  scale.Fit(input);
  scale.Transform(input, temp);
  CheckMatrices(input, temp, tolerance);
  scale.InverseTransform(input, temp);
  CheckMatrices(input, temp, tolerance);
}

/**
 * Test for Zero Scale.
 */
TEMPLATE_TEST_CASE("ZeroScaleTest", "[ScalingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "1 1 1 1;"
                    "2 6 10 18;";
  MatType scaled = "0 0 0 0;"
                   "0 0.2500 0.5000 1.000;";
  MatType scaleddataset, temp;
  MinMaxScaler<MatType> scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled, tolerance);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test for PCA whitening Scale.
 */
TEMPLATE_TEST_CASE("PCAWhiteningTest", "[ScalingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "1 1 1 1;"
                    "2 6 10 18;";
  PCAWhitening<MatType> scale;
  MatType output, temp;
  scale.Fit(dataset);
  scale.Transform(dataset, output);
  VecType diagonals = (ColumnCovariance(output)).diag();
  // Checking covarience is close to 1.0
  ElemType ccovsum = 0.0;
  for (size_t i = 0; i < diagonals.n_elem; ++i)
    ccovsum += diagonals(i);
  REQUIRE(ccovsum == Approx(1.0).epsilon(tolerance));
  scale.InverseTransform(output, temp);
  CheckMatrices(dataset, temp, tolerance);
}

/**
 * Test for ZCA whitening Scale.
 */
TEMPLATE_TEST_CASE("ZCAWhiteningTest", "[ScalingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;
  const double tolerance = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  MatType dataset = "1 1 1 1;"
                    "2 6 10 18;";
  ZCAWhitening<MatType> scale;
  MatType output, temp;
  scale.Fit(dataset);
  scale.Transform(dataset, output);
  VecType diagonals = (ColumnCovariance(output)).diag();
  // Check that the covariance is close to 1.0.
  ElemType ccovsum = 0.0;
  for (size_t i = 0; i < diagonals.n_elem; ++i)
    ccovsum += diagonals(i);
  REQUIRE(ccovsum == Approx(1.0).epsilon(tolerance));
  scale.InverseTransform(output, temp);
  CheckMatrices(dataset, temp, tolerance);
}
