
#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_DEV_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_DEV_H

#include "linear_regression_result.h"


void LinearRegressionResult::Init(const Matrix &query_table_in) {
  query_table_ = &query_table_in;
  predictions_.Init(query_table_->n_cols());
  residual_sum_of_squares_ = 0;
}


Vector &LinearRegressionResult::
predictions() {
  return predictions_;
}


const Vector &LinearRegressionResult::
predictions() const {
  return predictions_;
}


double LinearRegressionResult::residual_sum_of_squares() const {
  return residual_sum_of_squares_;
}


double &LinearRegressionResult::residual_sum_of_squares() {
  return residual_sum_of_squares_;
}

#endif
