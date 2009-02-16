#ifndef RIDGE_REGRESSION_H_
#define RIDGE_REGRESSION_H_
#include "fastlib/fastlib.h"

class RidgeRegression {
 public:
  RidgeRegression() {}
  void Init(fx_module *module, Matrix &predictors, Matrix &predictions);
  void Init(fx_module *module, Matrix &input_data, index_t selector);
  void Init(fx_module *module, 
            Matrix &input_data, 
            GenVector<index_t> &predictor_indices,
            index_t prediction_index);
  void Init(fx_module *module, 
            Matrix &input_data, 
            GenVector<index_t> &predictor_indices,
            Matrix &prediction);
  void Destruct();
  void Regress();
  double ComputeSquareError();
  void factors(Matrix *factors);
  void set_lambda(double lambda);
  double lambda();

 private:
  fx_module *module_;
  Matrix predictors_;
  Matrix predictions_;
  Matrix factors_;
  double lambda_;
  double lambda_sq_;
};

#include "ridge_regression_impl.h"
#endif

