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
#include <fastlib/base/test.h>
#include "ridge_regression.h"
#include "ridge_regression_util.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

class RidgeRegressionTest {
 public:
  void Init(fx_module *module) {
    module_ = module;
    arma::mat tmp;
    data::Load("predictors.csv", predictors_);
    data::Load("predictions.csv", predictions_);
    data::Load("true_factors.csv", true_factors_);
//    predictions_ = trans(predictions_);
    //predictors_ = trans(predictors_);
//    true_factors_ = trans(true_factors_);
    std::cout << predictors_ << '\n' <<
      predictions_ << '\n' <<
      true_factors_ << std::endl;
  }

  void TestSVDNormalEquationRegressVersusSVDRegress() {

    NOTIFY("[*] TestSVDNormalEquationRegressVersusSVDRegress");

    engine_ = new RidgeRegression();
    engine_->Init(module_, predictors_, predictions_, true);
    engine_->SVDRegress(0);
    RidgeRegression svd_engine;
    svd_engine.Init(module_, predictors_, predictions_, false);
    std::cout<< "wut" << std::endl;
    svd_engine.SVDRegress(0);
    arma::mat factors, svd_factors;
    engine_->factors(&factors);
    svd_engine.factors(&svd_factors);
    
    for(index_t i=0; i<factors.n_rows; i++) {
      NOTIFY("Normal Equation: %g, SVD: %g\n", factors(i, 0),
	     svd_factors(i, 0));
      TEST_DOUBLE_APPROX(factors(i, 0), svd_factors(i, 0), 1e-3);
    }
    
    Destruct();

    NOTIFY("[*] TestRegressVersusSVDRegress complete!");
  }

  void TestVIFBasedFeatureSelection() {
    
    NOTIFY("[*] TestVIFBasedFeatureSelection");

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

    engine_ = new RidgeRegression();
    engine_->Init(module_, synthetic_data, predictor_indices,
		  synthetic_data_target_training_values);
    engine_->FeatureSelectedRegression(predictor_indices,
				       prune_predictor_indices,
				       synthetic_data_target_training_values,
				       &output_predictor_indices);

    printf("Output indices: ");
    for(index_t i = 0; i < output_predictor_indices.n_elem; i++) {
      printf(" %d ", output_predictor_indices[i]);
    }
    printf("\n");
    NOTIFY("[*] TESTVIFBasedFeatureSelection complete!");
  }

  void TestAll() {
    TestSVDNormalEquationRegressVersusSVDRegress();
    TestVIFBasedFeatureSelection();
    NOTIFY("[*] All tests passed !!");
  }  

  void Destruct() {
    delete engine_;
  }

 private:
  fx_module *module_;
  RidgeRegression *engine_;
  arma::mat predictors_;
  arma::mat predictions_;
  arma::mat true_factors_;
};

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  fx_set_param_double(module, "lambda", 1.0);
  RidgeRegressionTest  test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}
