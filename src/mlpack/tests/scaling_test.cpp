/**
 * @file scaling_test.cpp
 *
 * Tests for Scaling of dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/pca_whitening.hpp>
#include <mlpack/core/data/scaler_methods/zca_whitening.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/scaler_methods/max_abs_scaler.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/scaler_methods/mean_normalization.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(ScalingTest);

arma::mat dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
arma::mat scaleddataset;
arma::mat temp;

/**
 * Test For MinMax Scaler Class.
 */
BOOST_AUTO_TEST_CASE(MinMaxScalerTest)
{
  arma::mat scaled = "0 0.2500 0.5000 1.000;"
                     "0 0.2500 0.5000 1.000;";
  data::MinMaxScaler scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled);
  CheckMatrices(dataset, temp);
}

/**
 * Test For MaxAbs Scaler Class.
 */
BOOST_AUTO_TEST_CASE(MaxAbsScalerTest)
{
  arma::mat scaled = "-1 -0.5 0 1;"
                     "0.1111111111 0.3333333333 0.55555556 1.0000;";
  data::MaxAbsScaler scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled);
  CheckMatrices(dataset, temp);
}

/**
 * Test For Standard Scaler Class.
 */
BOOST_AUTO_TEST_CASE(StandardScalerTest)
{
  arma::mat scaled = "-1.18321596 -0.50709255  0.16903085 1.52127766;"
                     "-1.18321596 -0.50709255  0.16903085 1.52127766;";
  data::StandardScaler scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled);
  CheckMatrices(dataset, temp);
}

/**
 * Test For MeanNormalization Scaler Class.
 */
BOOST_AUTO_TEST_CASE(MeanNormalizationTest)
{
  arma::mat scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                     "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
  data::MeanNormalization scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled);
  CheckMatrices(dataset, temp);
}

/**
 * Test to pass same matrix as input and output
 */
BOOST_AUTO_TEST_CASE(SameInputOutputTest)
{
  temp = dataset;
  arma::mat scaled = "-0.43750000000 -0.187500000 0.062500000 0.562500000;"
                     "-0.43750000000 -0.187500000 0.062500000 0.562500000;";
  data::MeanNormalization scale;
  scale.Fit(temp);
  scale.Transform(temp, temp);
  CheckMatrices(scaled, temp);
  scale.InverseTransform(temp, temp);
  CheckMatrices(dataset, temp);
}

/**
 * Test for Zero Matrix.
 */
BOOST_AUTO_TEST_CASE(ZeroMatrixTest)
{
  arma::mat input(2, 4, arma::fill::zeros);
  data::MeanNormalization scale;
  scale.Fit(input);
  scale.Transform(input, temp);
  CheckMatrices(input, temp);
  scale.InverseTransform(input, temp);
  CheckMatrices(input, temp);
}

/**
 * Test for Zero Scale.
 */
BOOST_AUTO_TEST_CASE(ZeroScaleTest)
{
  dataset = "1 1 1 1;"
            "2 6 10 18;";
  arma::mat scaled = "0 0 0 0;"
                     "0 0.2500 0.5000 1.000;";
  data::MinMaxScaler scale;
  scale.Fit(dataset);
  scale.Transform(dataset, scaleddataset);
  scale.InverseTransform(scaleddataset, temp);
  CheckMatrices(scaleddataset, scaled);
  CheckMatrices(dataset, temp);
}

/**
 * Test for PCA whitening Scale.
 */
BOOST_AUTO_TEST_CASE(PCAWhiteningTest)
{
  data::PCAWhitening scale;
  arma::mat output;
  scale.Fit(dataset);
  scale.Transform(dataset, output);
  arma::vec diagonals = (mlpack::math::ColumnCovariance(output)).diag();
  // Checking covarience is close to 1.0
  double ccovsum = 0.0;
  for (size_t i = 0; i < diagonals.n_elem; i++)
    ccovsum += diagonals(i);
  BOOST_REQUIRE_CLOSE(ccovsum, 1.0, 1e-3);
  scale.InverseTransform(output, temp);
  CheckMatrices(dataset, temp);
}

/**
 * Test for ZCA whitening Scale.
 */
BOOST_AUTO_TEST_CASE(ZCAWhiteningTest)
{
  data::ZCAWhitening scale;
  arma::mat output;
  scale.Fit(dataset);
  scale.Transform(dataset, output);
  arma::vec diagonals = (mlpack::math::ColumnCovariance(output)).diag();
  // Check that the covariance is close to 1.0.
  double ccovsum = 0.0;
  for (size_t i = 0; i < diagonals.n_elem; i++)
    ccovsum += diagonals(i);
  BOOST_REQUIRE_CLOSE(ccovsum, 1.0, 1e-3);
  scale.InverseTransform(output, temp);
  CheckMatrices(dataset, temp);
}

BOOST_AUTO_TEST_SUITE_END();
