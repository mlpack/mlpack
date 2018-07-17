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

//! Construct tree that rearranges the dataset
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    const typename std::enable_if<
        tree::TreeTraits<TreeType>::RearrangesDataset>::type* = 0)
{
  return new TreeType(std::forward<MatType>(dataset), oldFromNew);
}

//! Construct tree that doesn't rearrange the dataset
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename std::enable_if<
        !tree::TreeTraits<TreeType>::RearrangesDataset>::type* = 0)
{
  return new TreeType(std::forward<MatType>(dataset));
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::
KDE(const double bandwidth,
    const double relError,
    const double absError,
    const bool breadthFirst) :
    kernel(new KernelType(bandwidth)),
    metric(new MetricType()),
    relError(relError),
    absError(absError),
    breadthFirst(breadthFirst),
    ownsKernel(true),
    ownsMetric(true),
    ownsReferenceTree(false),
    trained(false)
{
  if (relError > 0 && absError > 0)
    Log::Warn << "Absolute and relative error tolerances will be sumed up"
              << std::endl;
  if (relError < 0 || absError < 0)
    Log::Fatal << "Error tolerance can't be less than 0" << std::endl;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::
KDE(MetricType& metric,
    KernelType& kernel,
    const double relError,
    const double absError,
    const bool breadthFirst) :
    kernel(&kernel),
    metric(&metric),
    relError(relError),
    absError(absError),
    breadthFirst(breadthFirst),
    ownsKernel(false),
    ownsMetric(false),
    ownsReferenceTree(false),
    trained(false)
{
  if (relError > 0 && absError > 0)
    Log::Warn << "Absolute and relative error tolerances will be sumed up"
              << std::endl;
  if (relError < 0 || absError < 0)
    Log::Fatal << "Error tolerance can't be less than 0" << std::endl;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::KDE(const KDE& other) :
    kernel(new KernelType(other.kernel)),
    metric(new MetricType(other.metric)),
    relError(other.relError),
    absError(other.absError),
    breadthFirst(other.breadthFirst),
    ownsKernel(other.ownsKernel),
    ownsMetric(other.ownsMetric),
    ownsReferenceTree(other.ownsReferenceTree),
    trained(other.trained)
{
  if (trained)
  {
    if (ownsReferenceTree)
      referenceTree = new Tree(other.referenceTree);
    else
      referenceTree = other.referenceTree;
  }
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::KDE(KDE&& other) :
    kernel(other.kernel),
    metric(other.metric),
    referenceTree(other.referenceTree),
    relError(other.relError),
    absError(other.absError),
    breadthFirst(other.breadthFirst),
    ownsKernel(other.ownsKernel),
    ownsMetric(other.ownsMetric),
    ownsReferenceTree(other.ownsReferenceTree),
    trained(other.trained)
{
  other.kernel = new KernelType();
  other.metric = new MetricType();
  other.referenceTree = nullptr;
  other.ownsReferenceTree = false;
  other.trained = false;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>&
KDE<MetricType, MatType, KernelType, TreeType>::operator=(KDE other)
{
  // Clean memory
  if (ownsKernel)
    delete kernel;
  if (ownsMetric)
    delete metric;
  if (ownsReferenceTree)
    delete referenceTree;

  // Move
  this->kernel = std::move(other.kernel);
  this->metric = std::move(other.metric);
  this->referenceTree = std::move(other.referenceTree);
  this->relError = other.relError;
  this->absError = other.absError;
  this->breadthFirst = other.breadthFirst;
  this->ownsKernel = other.ownsKernel;
  this->ownsMetric = other.ownsMetric;
  this->ownsReferenceTree = other.ownsReferenceTree;
  this->trained = other.trained;

  return *this;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
KDE<MetricType, MatType, KernelType, TreeType>::~KDE()
{
  if (ownsKernel)
    delete kernel;
  if (ownsMetric)
    delete metric;
  if (ownsReferenceTree)
    delete referenceTree;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
Train(const MatType& referenceSet)
{
  // Check if referenceSet is not an empty set.
  if (referenceSet.n_cols == 0)
    throw std::invalid_argument("cannot train KDE model with an empty "
                                "reference set");
  this->ownsReferenceTree = true;
  this->referenceTree = new Tree(referenceSet);
  this->trained = true;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
Train(Tree& referenceTree)
{
  // Check if referenceTree dataset is not an empty set.
  if (referenceTree.Dataset().n_cols == 0)
    throw std::invalid_argument("cannot train KDE model with an empty "
                                "reference set");
  if (this->ownsReferenceTree == true)
    delete this->referenceTree;
  this->ownsReferenceTree = false;
  this->referenceTree = &referenceTree;
  this->trained = true;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
Evaluate(const MatType& querySet, arma::vec& estimations)
{
  std::vector<size_t> oldFromNewQueries;
  Tree* queryTree = BuildTree<Tree>(querySet, oldFromNewQueries);
  typedef KDERules<MetricType, KernelType, Tree> RuleType;
  RuleType rules = RuleType(referenceTree->Dataset(),
                            queryTree->Dataset(),
                            estimations,
                            relError,
                            absError,
                            oldFromNewQueries,
                            *metric,
                            *kernel);
  if (breadthFirst)
  {
    // DualTreeTraverser Breadth-First
    typename Tree::template BreadthFirstDualTreeTraverser<RuleType>
      traverser(rules);
    traverser.Traverse(*queryTree, *referenceTree);
  }
  else
  {
    // DualTreeTraverser Depth-First
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);
    traverser.Traverse(*queryTree, *referenceTree);
  }
  estimations /= referenceTree->Dataset().n_cols;
  delete queryTree;

  // Ideas for the future...
  // SingleTreeTraverser
  /*
  typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);
  for(size_t i = 0; i < query.n_cols; ++i)
    traverser.Traverse(i, *referenceTree);
  */
  // Brute force
  /*
  arma::vec result = arma::vec(query.n_cols);
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
  return result;
  */
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
Evaluate(Tree& queryTree,
         const std::vector<size_t>& oldFromNewQueries,
         arma::vec& estimations)
{
  typedef KDERules<MetricType, KernelType, Tree> RuleType;
  RuleType rules = RuleType(referenceTree->Dataset(),
                            queryTree.Dataset(),
                            estimations,
                            relError,
                            absError,
                            oldFromNewQueries,
                            *metric,
                            *kernel);
  if (breadthFirst)
  {
    // DualTreeTraverser Breadth-First
    typename Tree::template BreadthFirstDualTreeTraverser<RuleType>
      traverser(rules);
    traverser.Traverse(queryTree, *referenceTree);
  }
  else
  {
    // DualTreeTraverser Depth-First
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);
    traverser.Traverse(queryTree, *referenceTree);
  }
  estimations /= referenceTree->Dataset().n_cols;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
RelativeError(const double newError)
{
  if (newError < 0 || newError > 1)
    Log::Fatal << "Relative error tolerance must be a value between 0 and 1"
               << std::endl;
  else
    this->relError = newError;
}

template<typename MetricType,
         typename MatType,
         typename KernelType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void KDE<MetricType, MatType, KernelType, TreeType>::
AbsoluteError(const double newError)
{
  if (newError < 0)
    Log::Fatal << "Absolute error tolerance must be a value greater or equal "
               << "to 0" << std::endl;
  else
    this->absError = newError;
}

} // namespace kde
} // namespace mlpack
