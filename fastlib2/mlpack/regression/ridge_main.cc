/*
 * =====================================================================================
 *
 *       Filename:  ridge_main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/16/2009 10:55:21 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  
 *
 * =====================================================================================
 */
#include <string>
#include "fastlib/fastlib.h"
#include "ridge_regression.h"

int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
  std::string predictors_file = fx_param_str_req(module,"predictors"); 
  std::string predictions_file = fx_param_str_req(module, "predictions");
  Matrix predictors;
  if (data::Load(predictors_file.c_str(), &predictors)==SUCCESS_FAIL) {
    FATAL("Unable to open file %s", predictors_file.c_str());
  }
  Matrix predictions;
  if (data::Load(predictions_file.c_str(), &predictions)==SUCCESS_FAIL) {
    FATAL("Unable to open file %s", predictions_file.c_str());
  }
  RidgeRegression engine;
  engine.Init(module, predictors, predictions);
  NOTIFY("Computing Regression...");
  engine.Regress();
  NOTIFY("Regression Complete !");
  double square_error =  engine.ComputeSquareError();
  NOTIFY("Squre Error:%g", square_error);
  fx_result_double(module, "square error", square_error);
  Matrix factors;
  engine.factors(&factors);
  std::string factors_file = fx_param_str(module, "factors", "factors.csv");
  NOTIFY("Saving factors...");
  data::Save(factors_file.c_str(), factors);  
  fx_done(module);
}
