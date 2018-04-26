/**
 * @file kde_impl.hpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Implementation of Kernel Density Estimation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "kde.hpp"
#include "kde_rules.hpp"
#include <cmath>

namespace mlpack {
namespace kde {

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::
KDE(const MatType& referenceSet,
    const double error,
    const double bandwidth) :
    referenceSet(referenceSet)
{
  this->referenceTree = new Tree(referenceSet);
  this->kernel = new KernelType(bandwidth);
  this->error = error;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::~KDE()
{
  delete this->referenceTree;
  delete this->kernel;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
Evaluate(const MatType& query, arma::vec& estimations)
{
  std::vector<size_t>* oldFromNewQueries;
  Tree* queryTree;
  oldFromNewQueries = new std::vector<size_t>(query.n_cols);
  queryTree = new Tree(query, *oldFromNewQueries);
  MetricType metric = MetricType();
  typedef KDERules<MetricType, KernelType, Tree> RuleType;
  RuleType rules = RuleType(this->referenceTree->Dataset(),
                            queryTree->Dataset(),
                            estimations,
                            error,
                            *oldFromNewQueries,
                            metric,
                            *kernel);
  // SingleTreeTraverser
  /*
  typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);
  for(size_t i = 0; i < query.n_cols; ++i)
    traverser.Traverse(i, *referenceTree);
  */

  // DualTreeTraverser
  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);
  estimations /= referenceSet.n_cols;
  delete oldFromNewQueries;
  delete queryTree;

  // Brute force
  /*arma::vec result = arma::vec(query.n_cols);
  result = arma::zeros<arma::vec>(query.n_cols);
  
  for(size_t i = 0; i < query.n_cols; ++i)
  {
    arma::vec density = arma::zeros<arma::vec>(referenceSet.n_cols);
    
    for(size_t j = 0; j < this->referenceSet.n_cols; ++j)
    {
      density(j) = this->kernel.Evaluate(query.col(i),
                                         this->referenceSet.col(j));
    }
    result(i) = arma::trunc_log(arma::sum(density)) -
      std::log(referenceSet.n_cols);
    //this->kernel.Normalizer(query.n_rows);
    //result(i) = (1/referenceSet.n_cols)*(accumulated);
  }
  return result;*/
}

} // namespace kde
} // namespace mlpack
