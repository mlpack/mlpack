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
  void Regress(double lambda);
  void SVDRegress(double lambda);
  void CrossValidatedRegression(double lambda_min, 
                               double lambda_max,
                               index_t num);
  double ComputeSquareError();
  void factors(Matrix *factors);

 private:
  fx_module *module_;
  Matrix predictors_;
  Matrix predictions_;
  Matrix factors_;

  void ComputeLinearModel_(double lambda_sq, const Vector &singular_values, 
			   const Matrix &u, const Matrix v_t);

};

#include "ridge_regression_impl.h"
#endif

