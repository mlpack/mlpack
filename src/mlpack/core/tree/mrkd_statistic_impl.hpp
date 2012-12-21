/**
 * @file mrkd_statistic_impl.hpp
 * @author James Cline
 *
 * Definition of the statistic for multi-resolution kd-trees.
 */
#ifndef __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP
#define __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP

// In case it hasn't already been included.
#include "mrkd_statistic.hpp"

namespace mlpack {
namespace tree {

/**
 * This constructor is called when a leaf is created.
 *
 * @param dataset Matrix that the tree is being built on.
 * @param begin Starting index corresponding to this leaf.
 * @param count Number of points held in this leaf.
 */
template<typename MatType>
MRKDStatistic::MRKDStatistic(const MatType& dataset,
                             const size_t begin,
                             const size_t count) :
    dataset(&dataset),
    begin(begin),
    count(count),
    leftStat(NULL),
    rightStat(NULL),
    parentStat(NULL)
{
  centerOfMass = dataset.col(begin);
  for (size_t i = begin+1; i < begin+count; ++i)
    centerOfMass += dataset.col(i);

  sumOfSquaredNorms = 0.0;
  for (size_t i = begin; i < begin+count; ++i)
    sumOfSquaredNorms += arma::norm(dataset.col(i), 2);
}

/**
 * This constructor is called when a non-leaf node is created.
 * This lets you build fast bottom-up statistics when building trees.
 *
 * @param dataset Matrix that the tree is being built on.
 * @param begin Starting index corresponding to this leaf.
 * @param count Number of points held in this leaf.
 * @param leftStat MRKDStatistic object of the left child node.
 * @param rightStat MRKDStatistic object of the right child node.
 */
template<typename MatType>
MRKDStatistic::MRKDStatistic(const MatType& dataset,
                             const size_t begin,
                             const size_t count,
                             MRKDStatistic& leftStat,
                             MRKDStatistic& rightStat) :
    dataset(&dataset),
    begin(begin),
    count(count),
    leftStat(&leftStat),
    rightStat(&rightStat),
    parentStat(NULL)
{
  sumOfSquaredNorms = leftStat.sumOfSquaredNorms + rightStat.sumOfSquaredNorms;

  /*
  centerOfMass = ((leftStat.centerOfMass * leftStat.count) +
                  (rightStat.centerOfMass * rightStat.count)) /
                  (leftStat.count + rightStat.count);
  */
  centerOfMass = leftStat.centerOfMass + rightStat.centerOfMass;

  isWhitelistValid = false;

  leftStat.parentStat = this;
  rightStat.parentStat = this;
}

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP
