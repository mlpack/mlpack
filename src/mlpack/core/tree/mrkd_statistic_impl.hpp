/**
 * @file mrkd_statistic_impl.hpp
 * @author James Cline
 *
 * Definition of the statistic for multi-resolution kd-trees.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP
#define __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP

// In case it hasn't already been included.
#include "mrkd_statistic.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
MRKDStatistic::MRKDStatistic(const TreeType& /* node */) :
    dataset(NULL),
    begin(0),
    count(0),
    leftStat(NULL),
    rightStat(NULL),
    parentStat(NULL)
{ }

/**
 * This constructor is called when a leaf is created.
 *
 * @param dataset Matrix that the tree is being built on.
 * @param begin Starting index corresponding to this leaf.
 * @param count Number of points held in this leaf.
 *
template<typename MatType>
MRKDStatistic::MRKDStatistic(const TreeType& node) :
    dataset(&dataset),
    begin(begin),
    count(count),
    leftStat(NULL),
    rightStat(NULL),
    parentStat(NULL)
{
  centerOfMass = dataset.col(begin);
  for (size_t i = begin + 1; i < begin + count; ++i)
    centerOfMass += dataset.col(i);

  sumOfSquaredNorms = 0.0;
  for (size_t i = begin; i < begin + count; ++i)
    sumOfSquaredNorms += arma::norm(dataset.col(i), 2);
}

 **
 * This constructor is called when a non-leaf node is created.
 * This lets you build fast bottom-up statistics when building trees.
 *
 * @param dataset Matrix that the tree is being built on.
 * @param begin Starting index corresponding to this leaf.
 * @param count Number of points held in this leaf.
 * @param leftStat MRKDStatistic object of the left child node.
 * @param rightStat MRKDStatistic object of the right child node.
 *
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

  *
  centerOfMass = ((leftStat.centerOfMass * leftStat.count) +
                  (rightStat.centerOfMass * rightStat.count)) /
                  (leftStat.count + rightStat.count);
  *
  centerOfMass = leftStat.centerOfMass + rightStat.centerOfMass;

  isWhitelistValid = false;

  leftStat.parentStat = this;
  rightStat.parentStat = this;
}
*/

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_MRKD_STATISTIC_IMPL_HPP
