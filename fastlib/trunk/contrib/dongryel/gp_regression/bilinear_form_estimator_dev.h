/** @author Dongryeol Lee
 *
 *  @file bilinear_form_estimator_dev.h
 */

#ifndef FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_BILINEAR_FORM_ESTIMATOR_DEV_H
#define FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_BILINEAR_FORM_ESTIMATOR_DEV_H

#include <vector>
#include "fastlib/la/matrix.h"
#include "fastlib/la/la.h"
#include "fastlib/la/uselapack.h"
#include "bilinear_form_estimator.h"

namespace fl {
namespace ml {

template<typename TransformationType>
void BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::PrintDebug(
  const char *name, FILE *stream) const {
  fprintf(stream, "----- MATRIX ------: %s\n", name);
  for (int r = 0; r < this->n_rows(); r++) {
    for (int c = 0; c < this->n_cols(); c++) {
      fprintf(stream, "%+3.3f ", this->get(r, c));
    }
    fprintf(stream, "\n");
  }
}

template<typename TransformationType>
int BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::n_rows() const {
  return diagonal_entries_->size();
}

template<typename TransformationType>
int BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::n_cols() const {
  return diagonal_entries_->size();
}

template<typename TransformationType>
double BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::get(int row, int col) const {
  if (row == col) {
    return (*diagonal_entries_)[row];
  }
  else if (row == col + 1 || col == row + 1) {
    return (*offdiagonal_entries_)[ std::min(row, col)];
  }
  else {
    return 0;
  }
}

template<typename TransformationType>
const std::vector<double> *BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::diagonal_entries() const {

  return diagonal_entries_;
}

template<typename TransformationType>
BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::TridiagonalLinearOperator(
  const std::vector<double> &diagonal_entries_in,
  const std::vector<double> &offdiagonal_entries_in) {

  diagonal_entries_ = &diagonal_entries_in;
  offdiagonal_entries_ = &offdiagonal_entries_in;
}

template<typename TransformationType>
int BilinearFormEstimator<TransformationType>::
TridiagonalLinearOperator::Apply(
  const Epetra_MultiVector &vecs,
  Epetra_MultiVector &prods) const {

  prods.PutScalar(0);

  for (int j = 0; j < diagonal_entries_->size(); j++) {

    for (int k = 0; k < vecs.NumVectors(); k++) {

      // Apply the diagonal entry.
      prods.Pointers()[k][j] += ((*diagonal_entries_)[j]) *
                                vecs.Pointers()[k][j];

      // Apply the lower diagonal entry.
      if (j > 0) {
        prods.Pointers()[k][j - 1] += ((*offdiagonal_entries_)[j - 1]) *
                                      vecs.Pointers()[k][j - 1];
      }

      // Apply the upper diagonal entry.
      if (j < diagonal_entries_->size() - 1) {
        prods.Pointers()[k][j + 1] += ((*offdiagonal_entries_)[j]) *
                                      vecs.Pointers()[k][j + 1];
      }
    }
  }
  return 0;
}

#ifdef EPETRA_MPI
template<typename TransformationType>
BilinearFormEstimator<TransformationType>::BilinearFormEstimator(): comm_(MPI_COMM_WORLD) {

}
#else
template<typename TransformationType>
BilinearFormEstimator<TransformationType>::BilinearFormEstimator() {

}
#endif

template<typename TransformationType>
Anasazi::LinearOperator *BilinearFormEstimator<TransformationType>::
linear_operator() {
  return linear_operator_;
}

template<typename TransformationType>
void BilinearFormEstimator<TransformationType>::Init(
  Anasazi::LinearOperator *linear_operator_in) {

  linear_operator_ = linear_operator_in;
  map_ = &(linear_operator_->OperatorDomainMap());
}

template<typename TransformationType>
double BilinearFormEstimator<TransformationType>::Dot_(
  const Epetra_MultiVector &first_vec,
  const Epetra_MultiVector &second_vec) const {

  double dot_product = 0;
  for (int i = 0; i < first_vec.GlobalLength(); i++) {
    dot_product += first_vec.Pointers()[0][i] * second_vec.Pointers()[0][i];
  }
  return dot_product;
}

template<typename TransformationType>
void BilinearFormEstimator<TransformationType>::AddExpert_(double scalar,
    const Epetra_MultiVector &source,
    Epetra_MultiVector *destination) const {
  for (int i = 0; i < source.GlobalLength(); i++) {
    destination->Pointers()[0][i] =
      destination->Pointers()[0][i] + scalar * source.Pointers()[0][i];
  }
}

template<typename TransformationType>
void BilinearFormEstimator<TransformationType>::Scale_(double scalar,
    const Epetra_MultiVector &source,
    Epetra_MultiVector *destination) const {
  for (int i = 0; i < source.GlobalLength(); i++) {
    destination->Pointers()[0][i] = scalar * source.Pointers()[0][i];
  }
}

template<typename TransformationType>
template<typename LinearOperatorType>
double BilinearFormEstimator<TransformationType>::ComputeQuadraticForm_(
  int num_iterations,
  const GenVector<double> &starting_vector,
  const LinearOperatorType &linear_operator_in,
  const Epetra_Map &map_in,
  int level,
  bool *break_down) {

  // The threshold for determining the convergence.
  const double convergence_threshold = 1e-7;

  // If it is a 1 by 1 matrix, then apply the transformation.
  if (linear_operator_in.n_rows() == 1 && linear_operator_in.n_cols() == 1) {
    return TransformationType::Transform(linear_operator_in.get(0, 0));
  }

  // The diagonal entries and the offdiagonal entries with the wrapper
  // class around it.
  std::vector<double> diagonal_entries;
  std::vector<double> offdiagonal_entries;
  TridiagonalLinearOperator tridiagonal_linear_operator(
    diagonal_entries, offdiagonal_entries);

  // The basis vector in the previous iteration.
  Epetra_MultiVector previous_vector(map_in, 1);
  previous_vector.PutScalar(0);

  // The basis vector in the current iteration.
  Epetra_MultiVector current_vector(map_in, 1);
  for (int i = 0; i < starting_vector.length(); i++) {
    current_vector.Pointers()[0][i] = starting_vector[i];
  }

  // A temporary vector used for matrix-vector multiplication.
  Epetra_MultiVector residual_vector(map_in, 1);
  residual_vector.PutScalar(0);

  // A temporary vector used for denoting the unit vector of varying
  // dimension.
  GenVector<double> unit_vector;
  unit_vector.Init(starting_vector.length());
  unit_vector.SetZero();
  unit_vector[0] = 1;

  // The old bilinear estimate.
  double old_bilinear_estimate = std::numeric_limits<double>::max();

  for (int j = 0; j < num_iterations; j++) {

    linear_operator_in.Apply(current_vector, residual_vector);
    if (j > 0) {
      AddExpert_(- (offdiagonal_entries[j - 1]),
                 previous_vector, &residual_vector);
    }

    // The dot product with the residual and the current basis
    // vector. Compute the off-diagonal and the diagonal entries
    // in this iteration.
    double alpha_j = Dot_(residual_vector, current_vector);
    diagonal_entries.push_back(alpha_j);
    AddExpert_(- alpha_j, current_vector, &residual_vector);
    double beta_j_plus_one =
      sqrt(Dot_(residual_vector, residual_vector));

    // Add in the offdiagonal entry.
    if (fabs(beta_j_plus_one) <= convergence_threshold) {
      *break_down = true;
      break;
    }
    offdiagonal_entries.push_back(beta_j_plus_one);

    // Take the current tridiagonal decomposition and estimate the
    // bilinear form. It is essential that Epetra_Map is constructed
    // here right before the recursive call.
    GenVector<double> unit_vector_alias;
    unit_vector_alias.Alias(unit_vector.ptr(),
                            diagonal_entries.size());
    Epetra_Map tridiagonal_linear_operator_map(
      unit_vector_alias.length(), 0, comm_);

    bool subcase_break_down = false;
    double bilinear_estimate =
      ComputeQuadraticForm_(unit_vector_alias.length(), unit_vector_alias,
                            tridiagonal_linear_operator,
                            tridiagonal_linear_operator_map, level + 1,
                            &subcase_break_down);

    // Check whether it converged.
    if (subcase_break_down) {
      *break_down = true;
      break;
    }
    if (fabs(old_bilinear_estimate - bilinear_estimate) <=
        convergence_threshold && false) {
      *break_down = false;
      break;
    }
    else {
      old_bilinear_estimate = bilinear_estimate;
    }

    // Update the previous vector and the next vector.
    for (int i = 0; i < current_vector.GlobalLength(); i++) {
      previous_vector.Pointers()[0][i] = current_vector.Pointers()[0][i];
    }
    Scale_(1.0 / beta_j_plus_one, residual_vector, &current_vector);
  }

  return old_bilinear_estimate;
}

template<typename TransformationType>
double BilinearFormEstimator<TransformationType>::Compute(
  const GenVector<double> &left_argument,
  const GenVector<double> &right_argument,
  bool naive_compute) {

  // Use the formula: $u^T f(A) v = 0.25 * (y^T f(A) y - z^T f(A) z )
  // $ where $y = u + v$ and $z = u - v$.
  GenVector<double> sum, difference;
  la::AddInit(
    left_argument, right_argument, &sum);
  la::SubInit(
    right_argument, left_argument, &difference);
  double estimate =
    (naive_compute) ?
    0.25 * (NaiveCompute(sum) - NaiveCompute(difference)) :
    0.25 * (Compute(sum) - Compute(difference));
  return estimate;
}

template<typename TransformationType>
double BilinearFormEstimator<TransformationType>::NaiveCompute(
  const GenVector<double> &argument) {

  // The naive estimate to return.
  double naive_bilinear_estimate = 0;

  // Compute the kernel matrix, and its eigendecomposition.
  GenMatrix<double> kernel_matrix;
  GenVector<double> eigenvalues;
  GenMatrix<double> eigenvectors;
  GenMatrix<double> eigenvectors_transposed;
  kernel_matrix.Init(linear_operator_->n_rows(), linear_operator_->n_cols());
  for (int j = 0; j < linear_operator_->n_cols(); j++) {
    for (int i = 0; i < linear_operator_->n_rows(); i++) {
      kernel_matrix.set(i, j, linear_operator_->get(i, j));
    }
  }
  la::SVDInit(
    kernel_matrix, &eigenvalues, &eigenvectors, &eigenvectors_transposed);
  eigenvalues.PrintDebug();

  // Project the argument to the eigenspace.
  GenVector<double> projected_vector;
  la::MulInit(
    eigenvectors_transposed, argument, &projected_vector);
  for (int i = 0; i < eigenvalues.length(); i++) {
    naive_bilinear_estimate += math::Sqr(projected_vector[i]) *
                               TransformationType::Transform(eigenvalues[i]);
  }
  return naive_bilinear_estimate;
}

template<typename TransformationType>
double BilinearFormEstimator<TransformationType>::Compute(
  const GenVector<double> &argument) {

  // Pass in the normalized unit vector to the quadratic form
  // computation and correct it afterwards.
  GenVector<double> normalized_argument;
  double length = la::LengthEuclidean(argument);
  if (length > 0) {
    la::ScaleInit(
      1.0 / length, argument, &normalized_argument);
  }
  else {
    normalized_argument.Copy(argument);
  }
  bool break_down = false;
  return math::Sqr(length) * ComputeQuadraticForm_(
           normalized_argument.length(),
           normalized_argument,
           *linear_operator_,
           *map_,
           0,
           &break_down);
}
};
};

#endif
