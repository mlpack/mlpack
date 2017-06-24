/**
 * @file parallel_sgd_impl.hpp
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
    arma::mat& dataset, arma::vec& labels) : dataset(dataset), labels(labels)
{
  numFunctions = dataset.n_cols;
}

double SparseSVMLossFunction::Evaluate(arma::mat& weights, size_t id)
{
  return std::max(0.0, 1 - labels(id) * arma::dot(weights, dataset.col(id)));
}

void SparseSVMLossFunction::Gradient(
    arma::mat& weights, size_t id, arma::mat& gradient)
{
  double dot = 1 - labels(id) * arma::dot(weights, dataset.unsafe_col(id));
  gradient = (dot < 0) ? arma::vec(weights.n_elem, arma::fill::zeros) : 
    (-1 * dataset.unsafe_col(id) * labels(id));
}
arma::Col<size_t> SparseSVMLossFunction::Components(size_t id)
{
  std::vector<size_t> nonZeroComponents;
  for(size_t i = 0; i < dataset.n_rows; ++i)
  {
    if(dataset(i, id) != 0.f)
    {
      nonZeroComponents.push_back(i);
    }
  }
  return arma::Col<size_t>(nonZeroComponents);
}

#endif
