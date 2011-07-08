/*
 * =====================================================================================
 *
 *       Filename:  ridge_regression_test.cc
 *
 *    Description: 
 *
 *        Version:  1.0
 *        Created:  02/15/2009 01:34:25 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <fastlib/fastlib.h>
#include "ridge_regression.h"
#include "ridge_regression_util.h"
#include <armadillo>
#include <fastlib/base/arma_compat.h>
#define BOOST_TEST_MODULE RidgeRegressionTest
#include <boost/test/unit_test.hpp>
using namespace mlpack;
BOOST_AUTO_TEST_CASE(TestSVDNormalEquationRegressVersusSVDRegress) {
   
    RidgeRegression engine_;
    arma::mat predictors_ = "1.2 4.2 2.1 0.3 4.2;"
                            "3.1 1.1 4.7 1.8 0.4;"
                            "2.5 3.3 9.1 7.4 0.1";
    arma::mat predictions_ = "0.4 0.33 0.8 1.4 3.3";
    arma::mat true_factors_ = "0; 0; 0";

    engine_.Init(predictors_, predictions_, true);
    engine_.SVDRegress(0);
    RidgeRegression svd_engine;
    svd_engine.Init(predictors_, predictions_, true);
    svd_engine.SVDRegress(0);
    arma::mat factors, svd_factors;
 
    engine_.factors(&factors);
    svd_engine.factors(&svd_factors);

    for(index_t i=0; i<factors.n_rows; i++) {
        BOOST_REQUIRE_CLOSE(factors(i, 0), svd_factors(i, 0), 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(TestVIFBasedFeatureSelection) {
    RidgeRegression engine_;
    
    // Craft a synthetic dataset in which the third dimension is
    // completely dependent on the first and the second.
    arma::mat synthetic_data;
    arma::mat synthetic_data_target_training_values;
    synthetic_data.zeros(4, 5);
    synthetic_data_target_training_values.zeros(1, 5);
    for(index_t i = 0; i < 5; i++) {
      synthetic_data(0, i) = i;
      synthetic_data(1, i) = 3 * i + 1;
      synthetic_data(2, i) = 4;
      synthetic_data(3, i) = 5;
      synthetic_data_target_training_values(0, i) = i;
    }
    arma::Col<index_t> predictor_indices;
    arma::Col<index_t> prune_predictor_indices;
    arma::Col<index_t> output_predictor_indices;
    predictor_indices.zeros(4);
    predictor_indices[0] = 0;
    predictor_indices[1] = 1;
    predictor_indices[2] = 2;
    predictor_indices[3] = 3;
    prune_predictor_indices = predictor_indices;
    engine_.Init(synthetic_data, predictor_indices,
                  synthetic_data_target_training_values);
    engine_.FeatureSelectedRegression(predictor_indices,
                                       prune_predictor_indices,
                                       synthetic_data_target_training_values,
                                       &output_predictor_indices);
    printf("Output indices: ");
    for(index_t i = 0; i < output_predictor_indices.n_elem; i++) {
      printf(" %"LI" ", output_predictor_indices[i]);
    }
    printf("\n");
}
