#ifndef FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_DEV_H
#define FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_DEV_H

#include "givens_rotate.h"

template<typename MatrixType>
void GivensRotate::ApplyToRow(double cosine_value, double sine_value,
                              int first_row_index,
                              int second_row_index,
                              MatrixType &matrix) {

  for (index_t j = 0; j < matrix.n_cols(); j++) {
    double first_row_number = matrix.get(first_row_index, j);
    double second_row_number = matrix.get(second_row_index, j);
    double first_new_number =
      cosine_value * first_row_number + sine_value * second_row_number;
    double second_new_number =
      -sine_value * first_row_number + cosine_value * second_row_number;
    matrix.set(first_row_index, j, first_new_number);
    matrix.set(second_row_index, j, second_new_number);
  }
}

template<typename MatrixType>
void GivensRotate::ApplyToColumn(double cosine_value, double sine_value,
                                 int first_column_index,
                                 int second_column_index,
                                 MatrixType &matrix) {

  for (index_t j = 0; j < matrix.n_rows(); j++) {
    double first_column_number = matrix.get(j, first_column_index);
    double second_column_number = matrix.get(j, second_column_index);
    double first_new_number =
      cosine_value * first_column_number - sine_value *
      second_column_number;
    double second_new_number =
      sine_value * first_column_number + cosine_value *
      second_column_number;
    matrix.set(j, first_column_index, first_new_number);
    matrix.set(j, second_column_index, second_new_number);
  }
}

/** @brief Computes the Givens rotation such that the second
 *         value becomes zero.
 */
void GivensRotate::Compute(double first, double second,
                           double *magnitude, double *cosine_value,
                           double *sine_value) {
  if (second == 0) {
    *magnitude = fabs(first);
    *cosine_value = (first >= 0) ? 1 : -1;
    *sine_value = 0;
  }
  else
    if (first == 0) {
      *magnitude = fabs(second);
      *cosine_value = 0;
      *sine_value = (second >= 0) ? 1 : -1;
    }
    else
      if (fabs(second) > fabs(first)) {
        double t = first / second;
        double u = (second >= 0) ? sqrt(1 + t * t) : -sqrt(1 + t * t);
        *magnitude = second * u;
        *sine_value = 1.0 / u;
        *cosine_value = (*sine_value) * t;
      }
      else {
        double t = second / first;
        double u = (first >= 0) ? sqrt(1 + t * t) : -sqrt(1 + t * t);
        *cosine_value = 1.0 / u;
        *sine_value = (*cosine_value) * t;
        *magnitude = first * u;
      }
}

#endif
