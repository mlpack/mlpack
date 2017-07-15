/**
 * @file sparse_svm_function_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of the hinge loss function for training a sparse SVM with the
 * parallel SGD algorithm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_svm_function.hpp"

SparseSVMFunction::SparseSVMFunction(
    const arma::sp_mat& dataset, const arma::vec& labels) :
  dataset(dataset), labels(labels)
{ /* Nothing to do */ }

double SparseSVMFunction::Evaluate(const arma::mat& parameters, size_t id)
{
  // The hinge loss function.
  return std::max(0.0, 1 - labels(id) * arma::dot(dataset.col(id), parameters));
}

template <typename GradType>
void SparseSVMFunction::Gradient(
    const arma::mat& parameters, size_t id, GradType& gradient)
{
  // Evaluate the gradient of the hinge loss function.
  double dot = 1 - labels(id) * arma::dot(parameters, dataset.col(id));
  gradient = (dot < 0) ? GradType(parameters.n_rows, 1) :
    (-1 * GradType(dataset.col(id) * labels(id)));
}

size_t SparseSVMFunction::NumFunctions()
{
  // The number of points in the dataset is the number of functions, as this
  // is a data dependent function.
  return dataset.n_cols;
}

#endif // MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_FUNCTION_IMPL_HPP
