#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_RESULT_H

#include "fastlib/la/matrix.h"

class LinearRegressionResult {
  private:

    const Matrix *query_table_;

    Vector predictions_;

    double residual_sum_of_squares_;

  public:

    void Init(const Matrix &query_table_in);

    const Vector &predictions() const;

    Vector &predictions();

    double residual_sum_of_squares() const;

    double &residual_sum_of_squares();
};

#endif
