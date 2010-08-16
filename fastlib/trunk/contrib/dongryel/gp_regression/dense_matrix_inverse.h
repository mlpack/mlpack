/** @file dense_matrix_inverse.h
 *
 *  @brief A utility for updating a dense matrix when its number of
 *         rows and number of columns grow by one.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_GP_REGRESSION_DENSE_MATRIX_INVERSE_H
#define ML_GP_REGRESSION_DENSE_MATRIX_INVERSE_H

#include "fastlib/fastlib.h"

namespace ml {
class DenseMatrixInverse {
  public:

    static Matrix *Update(
      const Matrix &previous_inverse,
      const Vector &inverse_times_new_column,
      double projection_error) {

      Matrix *new_matrix_inverse = new Matrix();
      new_matrix_inverse->Init(previous_inverse.n_rows() + 1,
                               previous_inverse.n_cols() + 1);

      for(int j = 0; j < previous_inverse.n_cols(); j++) {
        for(int i = 0; i < previous_inverse.n_rows(); i++) {
          new_matrix_inverse->set(
            i, j, previous_inverse.get(i, j) +
            inverse_times_new_column[i] *
            inverse_times_new_column[j] / projection_error);
        }
      }

      for(int j = 0; j < previous_inverse.n_cols(); j++) {
        new_matrix_inverse->set(
          j,
          previous_inverse.n_cols(), - inverse_times_new_column[j] /
          projection_error);
        new_matrix_inverse->set(
          previous_inverse.n_rows(), j,
          - inverse_times_new_column[j] / projection_error);
      }
      new_matrix_inverse->set(
        new_matrix_inverse->n_rows() - 1,
        new_matrix_inverse->n_cols() - 1,
        1.0 / projection_error);

      // Return the computed inverse.
      return new_matrix_inverse;
    }
};
};

#endif
