/*
 * =====================================================================================
 * 
 *       Filename:  ridge_regression_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/15/2009 12:19:22 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 * 
 * =====================================================================================
*/ 
#ifndef RIDGE_REGRESSION_H_
//COMPILER_PRINTF("%s", "!!!!!You have accidently included ridge_regression_impl.h "
//    "Fix it otherwise the program will behave unexpectedly");                
#else

void RidgeRegression::Init(fx_module *module, 
                           Matrix &predictors, 
                           Matrix &predictions) {
  module_=module;
  lambda_=fx_param_double(module_, "lambda", 0);
  lambda_sq_ = lambda_ * lambda_;
  DEBUG_ERROR_MSG_IF(predictors.n_cols()<predictors.n_rows(),
     "The number of the columns %"LI"d must be less or equal to the number of "
     " the rows %"LI"d ", predictors.n_cols(), predictors.n_rows());
  DEBUG_ERROR_MSG_IF(predictions.n_rows() >1, 
      "The current implementation supports only one dimensional predictions");
  DEBUG_ERROR_MSG_IF(predictors.n_cols()!=predictions.n_cols(), 
      "Predictors and predictions must have the same same number "
      "of rows %"LI"d != %"LI"d ", predictors.n_cols(), predictions.n_cols());
  predictors_.Copy(predictors);
  predictions_.Copy(predictions);
}
void RidgeRegression::Init(fx_module *module, 
                           Matrix &input_data, 
                           index_t selector) {
  //COMPILER_PRINTF("%s","!!!!!!!!!! Method not implemented yet" );
}
void RidgeRegression::Init(fx_module *module, 
                           Matrix &input_data, 
                           GenVector<index_t> &predictor_indices,
                           index_t prediction_index) {
  //COMPILER_PRINTF("%s","!!!!!!!!!! Method not implemented yet" );

}
void RidgeRegression::Init(fx_module *module, 
                           Matrix &input_data, 
                           GenVector<index_t> &predictor_indices,
                           Matrix &prediction) {
  //COMPILER_PRINTF("%s","!!!!!!!!!! Method not implemented yet" );

}

void RidgeRegression::Regress() {
  // we have to solve the system:
  // (predictors * predictors^T + lambda^2 I) * factors 
  //     = predictors^T * predictions 
  Matrix lhs; // lhs: left hand side
  Matrix rhs; // rhs: right hand side
 
  la::MulTransBInit(predictors_, predictors_, &lhs);
  for(index_t i=0; i<predictors_.n_rows(); i++) {
    lhs.set(i, i, lhs.get(i, i) + lambda_sq_);
  }
  la::MulTransBInit(predictors_, predictions_, &rhs);
  la::SolveInit(lhs, rhs, &factors_);
}

double RidgeRegression::ComputeSquareError() {
  Matrix error;
  la::MulTransAInit(factors_, predictors_, &error);
  la::SubFrom(error, &predictions_);
  double square_error= la::Dot(error.n_rows(), error.ptr(), error.ptr()); 
  return square_error;
}

void  RidgeRegression::Factors(Matrix *factors) {
  factors->Copy(factors_);
}
  
void  RidgeRegression::set_lambda(double lambda) {
  lambda_ = lambda;
  lambda_sq_ = lambda_ * lambda_;
}
double  RidgeRegression::lambda() {
  return lambda_;
}


#endif
