/**
 * @file ridge_regression_test.cpp
 *
 * Test for ridge regression.
 */
#include <mlpack/core.h>
#include <mlpack/methods/regression/ridge_regression.h>
#include <mlpack/methods/regression/ridge_regression_util.h>

#include <iostream>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RidgeRegressionTest);

/*
BOOST_AUTO_TEST_CASE(TestSVDNormalEquationRegressVersusSVDRegress)
{
  arma::mat predictors_ = "1.2 4.2 2.1 0.3 4.2;"
                          "3.1 1.1 4.7 1.8 0.4;"
                          "2.5 3.3 9.1 7.4 0.1";
  arma::mat predictions_ = "0.4 0.33 0.8 1.4 3.3";
  arma::mat true_factors_ = "0; 0; 0";

  RidgeRegression engine_(predictors_, predictions_, true);
  engine_.SVDRegress(0);
  RidgeRegression svd_engine(predictors_, predictions_, false);
  svd_engine.SVDRegress(0);
  arma::mat factors, svd_factors;

  engine_.factors(&factors);
  svd_engine.factors(&svd_factors);

  for(size_t i=0; i<factors.n_rows; i++)
    BOOST_REQUIRE_CLOSE(factors(i, 0), svd_factors(i, 0), 1e-5);
}
 */

BOOST_AUTO_TEST_CASE(TestVIFBasedFeatureSelection)
{
  // Craft a synthetic dataset in which the third dimension is
  // completely dependent on the first and the second.
  arma::mat synthetic_data;
  arma::mat synthetic_data_target_training_values;
  synthetic_data.zeros(4, 5);
  synthetic_data_target_training_values.zeros(1, 5);

  for(size_t i = 0; i < 5; i++)
  {
    synthetic_data(0, i) = i;
    synthetic_data(1, i) = 3 * i + 1;
    synthetic_data(2, i) = 4;
    synthetic_data(3, i) = 5;
    synthetic_data_target_training_values(0, i) = i;
  }

  arma::Col<size_t> predictor_indices;
  arma::Col<size_t> prune_predictor_indices;
  arma::Col<size_t> output_predictor_indices;
  predictor_indices.zeros(4);
  predictor_indices[0] = 0;
  predictor_indices[1] = 1;
  predictor_indices[2] = 2;
  predictor_indices[3] = 3;
  prune_predictor_indices = predictor_indices;
  RidgeRegression engine_(synthetic_data, predictor_indices,
      synthetic_data_target_training_values);
  engine_.FeatureSelectedRegression(predictor_indices,
      prune_predictor_indices,
      synthetic_data_target_training_values,
      &output_predictor_indices);
  std::cout << "Output indices: ";
  for(size_t i = 0; i < output_predictor_indices.n_elem; i++)
    std::cout << output_predictor_indices[i] << ' ';
  std::cout << std::endl;
}

BOOST_AUTO_TEST_SUITE_END();
