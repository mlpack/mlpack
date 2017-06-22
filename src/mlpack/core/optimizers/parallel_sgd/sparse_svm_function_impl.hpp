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
  GenerateVisitationOrder();
}

void SparseSVMLossFunction::GenerateVisitationOrder()
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (numFunctions - 1), numFunctions));
}

arma::Col<size_t> SparseSVMLossFunction::VisitationOrder(
    size_t thread_id, size_t max_threads)
{
  arma::Col<size_t> threadShare;
  if (thread_id == max_threads - 1){
    // The last thread gets the remaining instances
    threadShare = visitationOrder.subvec(thread_id * (numFunctions /
        max_threads), numFunctions - 1);
  }
  else 
  {
    // An equal distribution of data
    threadShare = visitationOrder.subvec(thread_id * (numFunctions /
        max_threads), (thread_id + 1) * (numFunctions / max_threads) - 1);
  }
  return threadShare;
}

arma::vec SparseSVMLossFunction::Gradient(
    arma::mat& weights, size_t id)
{
  double dot = 1 - labels(id) * arma::dot(weights, dataset.unsafe_col(id));
  return (dot < 0) ? arma::vec(weights.n_elem, arma::fill::zeros) : 
    -1 * weights labels(id);
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
