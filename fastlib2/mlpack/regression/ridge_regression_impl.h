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

void RidgeRegression::Regress(double lambda) {
  // we have to solve the system:
  // (predictors * predictors^T + lambda^2 I) * factors 
  //     = predictors^T * predictions 
  Matrix lhs; // lhs: left hand side
  Matrix rhs; // rhs: right hand side
  double lambda_sq = lambda * lambda;
  la::MulTransBInit(predictors_, predictors_, &lhs);
  for(index_t i=0; i<predictors_.n_rows(); i++) {
    lhs.set(i, i, lhs.get(i, i) + lambda_sq);
  }
  la::MulTransBInit(predictors_, predictions_, &rhs);
  la::SolveInit(lhs, rhs, &factors_);
}

void RidgeRegression::SVDRegress(double lambda) {
  Vector singular_values;
  Matrix u, v_t;
  Matrix predictors_t;
  la::TransposeInit(predictors_, &predictors_t);
  la::SVDInit(predictors_t, &singular_values, &u, &v_t);
  factors_.Init(1, predictors_.n_cols());
  factors_.SetAll(0.0);
  double lambda_sq = lambda * lambda;
  for(index_t i = 0; i < singular_values.length(); i++) {
    double s_sq = math::Sqr(singular_values[i]);
    double alpha = s_sq / (lambda_sq + s_sq) *
      la::Dot(predictions_.n_cols(), u.GetColumnPtr(i), predictions_.ptr());
    la::AddExpert(factors_.n_cols(), alpha, u.GetColumnPtr(i),
		  factors_.ptr());   
  }
}

void RidgeRegression::CrossValidatedRegression(double lambda_min, 
					       double lambda_max,
					       index_t num) {
  DEBUG_ERROR_MSG_IF(lambda_min>lambda_max, 
      "lambda_max %lg must be larger than lambda_min %lg",
     lambda_max, lambda_min );
  double step=(lambda_max-lambda_min)/num;
  Vector singular_values;
  Matrix u, v_t;
  Matrix predictors_t;
  la::TransposeInit(predictors_, &predictors_t);
  la::SVDInit(predictors_t, &singular_values, &u, &v_t);
  Vector singular_values_sq;
  singular_values_sq.Copy(singular_values);
  for(index_t i=0; i<singular_values.length(); i++) {
    singular_values_sq[i] = math::Sqr(singular_values[i]);
  }
  Matrix u_x_b;
  la::MulInit(u, predictions_, &u_x_b);
  double min_score=DBL_MAX;
  index_t min_index=-1;
  for(index_t i=0; i<num; i++) {
    double lambda=lambda_min+i*step;
    double lambda_sq=math::Sqr(lambda);
    // compute residual error
    Matrix error;
    error.Init(1, predictors_.n_cols());
    double tau=predictors_.n_rows();
    for(index_t j=0; i<singular_values_sq.length(); j++) {
      double alpha=lambda_sq/(singular_values_sq[j]+lambda_sq);
      la::AddExpert(error.n_cols(), 
                    alpha, 
                    u_x_b.GetColumnPtr(i), 
                    error.ptr());
      // compute tau
      tau-=singular_values_sq[j]/(singular_values_sq[j]+lambda_sq);
    }
    double rss = la::Dot(error, error);
    double score=(rss)/math::Sqr(tau);
    if (score<min_score) {
      min_score=score;
      min_index=i;
    }
  }
  fx_result_double(module_, "cross_validation_score", min_score);
  double lambda_sq = math::Sqr(lambda_min+min_index*step);
  factors_.Init(1, predictors_.n_cols());
  for(index_t i=0; i<singular_values_sq.length(); i++) {
      double alpha = singular_values_sq[i] / 
	(singular_values_sq[i] + lambda_sq);
      la::AddExpert(factors_.n_cols(), alpha, u_x_b.GetColumnPtr(i), 
                    factors_.ptr());
   }

}


double RidgeRegression::ComputeSquareError() {
  Matrix error;
  la::MulTransAInit(factors_, predictors_, &error);
  la::SubFrom(error, &predictions_);
  double square_error= la::Dot(error.n_rows(), error.ptr(), error.ptr()); 
  return square_error;
}

void  RidgeRegression::factors(Matrix *factors) {
  factors->Copy(factors_);
}

#endif
