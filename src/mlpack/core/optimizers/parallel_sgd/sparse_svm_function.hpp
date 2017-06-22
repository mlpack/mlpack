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
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_SVM_HPP
#include <mlpack/prereqs.hpp>

class SparseSVMLossFunction{
 public:
  SparseSVMLossFunction(arma::mat& dataset, arma::vec& labels);
  double Evaluate(arma::mat &weights);
  arma::vec Gradient(arma::mat& weights, size_t component);
  arma::Col<size_t> VisitationOrder(size_t thread_id, size_t max_threads);
  arma::Col<size_t> Components(size_t id);
  void GenerateVisitationOrder();
 private:
  arma::mat dataset;
  arma::vec labels;
  arma::Col<size_t> visitationOrder;
  size_t numFunctions;
};

// Include implementation
#include "sparse_svm_function_impl.hpp"

#endif
