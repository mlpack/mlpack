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

void RidgeRegression::Init(fx_module *module, Matrix &predictors, 
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

  // Append a row of 1's to the predictors to account for dataset that
  // is not mean-centered, and transpose it.
  predictors_.Init(predictions.n_cols(), predictors.n_rows() + 1);
  for(index_t i = 0; i < predictors_.n_rows(); i++) {
    predictors_.set(i, 0, 1.0);
    for(index_t j = 1; j < predictors_.n_cols(); j++) {
      predictors_.set(i, j, predictors.get(j - 1, i));
    }
  }
  la::TransposeInit(predictions, &predictions_);
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

  // we have to solve the system: (predictors^T * predictors +
  // lambda^2 I) * factors = predictors^T * predictions
  Matrix lhs; // lhs: left hand side
  Matrix rhs; // rhs: right hand side
  double lambda_sq = lambda * lambda;
  la::MulTransAInit(predictors_, predictors_, &lhs);

  // Add the regularizaton parameter to the diagonals before
  // inverting.
  for(index_t i = 0; i < predictors_.n_cols(); i++) {
    lhs.set(i, i, lhs.get(i, i) + lambda_sq);
  }

  la::MulTransAInit(predictors_, predictions_, &rhs);

  success_t flag = la::SolveInit(lhs, rhs, &factors_);

  if(flag == SUCCESS_FAIL) {
    printf("There was a problem in inverting the matrix!\n");
  }
}

void RidgeRegression::ComputeLinearModel_
(double lambda_sq, const Vector &singular_values, 
 const Matrix &u, const Matrix v_t) {

  // Factors should have $D + 1$ parameters.
  factors_.Init(predictors_.n_cols(), 1);
  factors_.SetZero();

  for(index_t i = 0; i < singular_values.length(); i++) {
    double s_sq = math::Sqr(singular_values[i]);
    double alpha = singular_values[i] / (lambda_sq + s_sq) * 
      la::Dot(u.n_rows(), u.GetColumnPtr(i), predictions_.ptr());

    // Scale each row vector of V^T and add to the factor.
    for(index_t j = 0; j < v_t.n_cols(); j++) {
      factors_.set(j, 0, factors_.get(j, 0) + alpha * v_t.get(i, j));
    }
  }
}

void RidgeRegression::SVDRegress(double lambda) {

  Vector singular_values;
  Matrix u, v_t;
  la::SVDInit(predictors_, &singular_values, &u, &v_t);

  double lambda_sq = lambda * lambda;

  ComputeLinearModel_(lambda_sq, singular_values, u, v_t);

  /*
  for(index_t i = 0; i < singular_values.length(); i++) {
    double s_sq = math::Sqr(singular_values[i]);
    double alpha = singular_values[i] / (lambda_sq + s_sq) * 
      la::Dot(u.n_rows(), u.GetColumnPtr(i), predictions_.ptr());

    // Scale each row vector of V^T and add to the factor.
    for(index_t j = 0; j < v_t.n_cols(); j++) {
      factors_.set(j, 0, factors_.get(j, 0) + alpha * v_t.get(i, j));
    }
  }
  */
}

void RidgeRegression::CrossValidatedRegression(double lambda_min, 
					       double lambda_max,
					       index_t num) {
  DEBUG_ERROR_MSG_IF(lambda_min > lambda_max, 
		     "lambda_max %lg must be larger than lambda_min %lg",
		     lambda_max, lambda_min );
  double step = (lambda_max - lambda_min) / num;
  Vector singular_values;
  Matrix u, v_t;
  la::SVDInit(predictors_, &singular_values, &u, &v_t);

  // Square the singular values and store it.
  Vector singular_values_sq;
  singular_values_sq.Copy(singular_values);
  for(index_t i = 0; i < singular_values.length(); i++) {
    singular_values_sq[i] = math::Sqr(singular_values[i]);
  }

  // u_x_b will be a vector of length s such that each entry is a dot
  // product between the $i$-th left singular vector and the
  // predictions_ values.
  Matrix u_x_b;
  la::MulTransAInit(u, predictions_, &u_x_b);
  double min_score = DBL_MAX;
  index_t min_index = -1;

  Matrix error;
  error.Init(1, predictors_.n_cols());

  // Try different values of lambda and choose the best one that
  // minimizes the loss function.
  for(index_t i = 0; i < num; i++) {
    double lambda = lambda_min + i * step;
    double lambda_sq = math::Sqr(lambda);

    // compute residual error
    error.SetZero();

    // tau starts from the number of columns of predictors_ minus one
    // because we append a column of 1's at the start to the
    // dimensionality of the problem.
    double tau = predictors_.n_cols() - 1;
    for(index_t j = 0; j < singular_values_sq.length(); j++) {
      double alpha = lambda_sq / (singular_values_sq[j] + lambda_sq);
      la::AddExpert(error.n_cols(), alpha * u_x_b.get(j, 0), 
                    u.GetColumnPtr(j), error.ptr());
      // compute tau
      tau -= singular_values_sq[j] / (singular_values_sq[j] + lambda_sq);
    }
    double rss = la::Dot(error, error);

    // Here we need to add to residual squared error the squared error
    // of the predictions.
    for(index_t j = 0; j < predictions_.n_rows(); j++) {
      double accumulant = predictions_.get(j, 0);
      
      for(index_t k = 0; k < singular_values_sq.length(); k++) {
	accumulant -= u_x_b.get(k, 0) * u.get(j, k);
      }
      rss += math::Sqr(accumulant);
    }

    double score = rss / math::Sqr(tau);
    if(score < min_score) {
      min_score = score;
      min_index = i;
    }
  }
  fx_result_double(module_, "cross_validation_score", min_score);

  // Using the best lambda, compute the linear model.
  double lambda_sq = math::Sqr(lambda_min + min_index * step);
  ComputeLinearModel_(lambda_sq, singular_values, u, v_t);
}


double RidgeRegression::ComputeSquareError() {
  Matrix error;
  la::MulInit(predictors_, factors_, &error);
  la::SubFrom(predictions_, &error);
  double square_error = la::Dot(error.n_rows(), error.ptr(), error.ptr());
  return square_error;
}

void  RidgeRegression::factors(Matrix *factors) {
  factors->Copy(factors_);
}

#endif
