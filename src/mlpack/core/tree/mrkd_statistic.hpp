/**
 * @file mrkd_statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 *
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP
#define __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP

namespace mlpack {
namespace tree {

/**
 * Statistic for multi-resolution kd-trees.
 */
class MRKDStatistic
{
  public:
    MRKDStatistic()
    :
      dataset(NULL),
      begin(0),
      count(0)
    { }

    ~MRKDStatistic() {}

    /**
     * This constructor is called when a leaf is created.
     *
     * @param dataset Matrix that the tree is being built on.
     * @param begin Starting index corresponding to this leaf.
     * @param count Number of points held in this leaf.
     */
    template<typename MatType>
    MRKDStatistic(const MatType& dataset,
                   const size_t begin,
                   const size_t count)
    :
      dataset(&dataset),
      begin(begin),
      count(count),
      parentStat(NULL)
    {
      centerOfMass = dataset.col(begin);
      for(size_t i = begin+1; i < begin+count; ++i)
        centerOfMass += dataset.col(i);

      sumOfSquaredNorms = 0.0;
      for(size_t i = begin; i < begin+count; ++i)
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
    MRKDStatistic(const MatType& dataset,
                   const size_t begin,
                   const size_t count,
                   MRKDStatistic& leftStat,
                   MRKDStatistic& rightStat)
    :
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

      isWhiteListValid = false;

      leftStat.parentStat = this;
      rightStat.parentStat = this;
    }

    //! The data points this object contains
    const arma::mat* dataset;
    //! The initial item in the dataset, so we don't have to make a copy
    size_t begin;
    //! The number of items in the dataset
    size_t count;
    //! The left child 
    const MRKDStatistic* leftStat;
    //! The right child 
    const MRKDStatistic* rightStat;
    //! A link to my parent node, null if I am the root
    const MRKDStatistic* parentStat;

    // Computed statistics
    //! The center of mass for this dataset
    arma::colvec centerOfMass;
    //! The sum of the squared Euclidian norms for this dataset
    double sumOfSquaredNorms;
		
		// There may be a better place to store this -- HRectBound?
		//! The index of the dominating centroid of the associated hyperrectangle
		size_t dominatingCentroid;

    //! The list of centroids that cannot own this hyperrectangle
    std::vector<size_t> whiteList;
    //! Whether or not the whitelist is valid
    bool isWhiteListValid;
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP
