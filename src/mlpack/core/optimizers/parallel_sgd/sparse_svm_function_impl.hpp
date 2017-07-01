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
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_svm_function.hpp"

SparseSVMLossFunction::SparseSVMLossFunction(
    arma::sp_mat& dataset, arma::vec& labels) :
  dataset(dataset), labels(labels)
{ /* Nothing to do */ }

double SparseSVMLossFunction::Evaluate(const arma::mat& weights, size_t id)
{
  return std::max(0.0, 1 - labels(id) * arma::dot(dataset.col(id), weights));
}

void SparseSVMLossFunction::Gradient(
    const arma::mat& weights, size_t id, arma::sp_mat& gradient)
{
  double dot = 1 - labels(id) * arma::dot(weights, dataset.col(id));
  gradient = (dot < 0) ? arma::sp_mat(weights.n_rows, 1) :
    (-1 * arma::sp_mat(dataset.col(id) * labels(id)));
}

size_t SparseSVMLossFunction::NumFunctions()
{
  return dataset.n_cols;
}

#endif
