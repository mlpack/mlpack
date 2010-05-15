/** @author Dongryeol Lee
 *
 *  @file log_determinant_dev.h
 */

#ifndef FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_LOG_DETERMINANT_DEV_H
#define FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_LOG_DETERMINANT_DEV_H

#include "fastlib/la/matrix.h"
#include "bilinear_form_estimator_dev.h"
#include "log_determinant.h"

namespace fl {
namespace ml {
LogDeterminant::LogDeterminant() {
}

void LogDeterminant::Init(Anasazi::LinearOperator *linear_operator_in) {
  bilinear_log_form_.Init(linear_operator_in);
}

void LogDeterminant::RandomVector_(GenVector<double> &v) {
  for (int i = 0; i < v.length(); i++) {
    v[i] = (math::Random() >= 0.5) ? 1 : -1;
  }
}

double LogDeterminant::MonteCarloCompute() {

  // A random vector for the samples.
  GenVector<double> random_vector;
  random_vector.Init(bilinear_log_form_.linear_operator()->n_rows());

  double log_determinant = 0;
  const int num_samples = 100;

  for (int i = 0; i < num_samples; i++) {
    RandomVector_(random_vector);
    log_determinant += bilinear_log_form_.Compute(random_vector);
  }
  log_determinant /= ((double) num_samples);
  return log_determinant;
}

double LogDeterminant::NaiveCompute() {

  double log_determinant = 0;
  GenMatrix<double> kernel_matrix;
  kernel_matrix.Init(bilinear_log_form_.linear_operator()->n_rows(),
                     bilinear_log_form_.linear_operator()->n_cols());
  for (int j = 0; j < bilinear_log_form_.linear_operator()->n_cols(); j++) {
    for (int i = 0; i < bilinear_log_form_.linear_operator()->n_rows(); i++) {
      kernel_matrix.set(i, j, bilinear_log_form_.linear_operator()->get(i, j));
    }
  }
  GenVector<double> eigenvalues;
  la::SVDInit(kernel_matrix, &eigenvalues);

  for (int i = 0; i < eigenvalues.length(); i++) {
    log_determinant += log(eigenvalues[i]);
  }
  return log_determinant;
}

double LogDeterminant::Compute() {

  // Do a naive for-loop over each row and apply $e_i^T log(A) e_i$.
  GenVector<double> i_th_unit_vector;
  i_th_unit_vector.Init(bilinear_log_form_.linear_operator()->n_rows());
  i_th_unit_vector.SetZero();
  double log_determinant = 0;

  for (int i = 0; i < bilinear_log_form_.linear_operator()->n_rows(); i++) {
    i_th_unit_vector[i] = 1;
    if (i > 0) {
      i_th_unit_vector[i - 1] = 0;
    }
    log_determinant += bilinear_log_form_.Compute(i_th_unit_vector);
  }
  return log_determinant;
}
};
};

#endif
