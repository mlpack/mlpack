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
#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "ridge_regression.h"

class RidgeRegressionTest {
 public:
  void Init(fx_module *module) {
    module_=module;
    data::Load("predictors.csv", &predictors_);
    data::Load("predictions.csv", &predictions_);
    data::Load("true_factors.csv", &true_factors_);
    engine_.Init(module_, predictors_, predictions_); 
  }
  void Test1() {
    engine_.Regress();
    Matrix factors;
    engine_.factors(&factors);
    NOTIFY("Square Error:%g", engine_.ComputeSquareError());
    for(index_t i=0; i<factors.n_rows(); i++) {
      TEST_DOUBLE_APPROX(factors.get(i,0), true_factors_.get(i, 0), 1e-3);
    }
  }
  void TestAll() {
    Test1();
    NOTIFY("[*] Test1 passed !!");
  }  
  void Destruct();
 private:
  fx_module *module_;
  RidgeRegression engine_;
  Matrix predictors_;
  Matrix predictions_;
  Matrix true_factors_;

};

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  fx_set_param_double(module, "lambda", 1.0);
  RidgeRegressionTest  test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}
