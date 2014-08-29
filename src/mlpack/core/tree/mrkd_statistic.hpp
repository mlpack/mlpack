/**
 * @file mrkd_statistic.hpp
 * @author James Cline
 *
 * Definition of the statistic for multi-resolution kd-trees.
 *
 * This file is part of MLPACK 1.0.10.
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

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * Statistic for multi-resolution kd-trees.
 */
class MRKDStatistic
{
 public:
  //! Initialize an empty statistic.
  MRKDStatistic();

  /**
   * This constructor is called when a node is finished initializing.
   *
   * @param node The node that has been finished.
   */
  template<typename TreeType>
  MRKDStatistic(const TreeType& /* node */);

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

  //! Get the index of the initial item in the dataset.
  size_t Begin() const { return begin; }
  //! Modify the index of the initial item in the dataset.
  size_t& Begin() { return begin; }

  //! Get the number of items in the dataset.
  size_t Count() const { return count; }
  //! Modify the number of items in the dataset.
  size_t& Count() { return count; }

  //! Get the center of mass.
  const arma::colvec& CenterOfMass() const { return centerOfMass; }
  //! Modify the center of mass.
  arma::colvec& CenterOfMass() { return centerOfMass; }

  //! Get the index of the dominating centroid.
  size_t DominatingCentroid() const { return dominatingCentroid; }
  //! Modify the index of the dominating centroid.
  size_t& DominatingCentroid() { return dominatingCentroid; }

  //! Access the whitelist.
  const std::vector<size_t>& Whitelist() const { return whitelist; }
  //! Modify the whitelist.
  std::vector<size_t>& Whitelist() { return whitelist; }

 private:
  //! The data points this object contains.
  const arma::mat* dataset;
  //! The initial item in the dataset, so we don't have to make a copy.
  size_t begin;
  //! The number of items in the dataset.
  size_t count;
  //! The left child.
  const MRKDStatistic* leftStat;
  //! The right child.
  const MRKDStatistic* rightStat;
  //! A link to the parent node; NULL if this is the root.
  const MRKDStatistic* parentStat;

  // Computed statistics.
  //! The center of mass for this dataset.
  arma::colvec centerOfMass;
  //! The sum of the squared Euclidean norms for this dataset.
  double sumOfSquaredNorms;

  // There may be a better place to store this -- HRectBound?
  //! The index of the dominating centroid of the associated hyperrectangle.
  size_t dominatingCentroid;

  //! The list of centroids that cannot own this hyperrectangle.
  std::vector<size_t> whitelist;
  //! Whether or not the whitelist is valid.
  bool isWhitelistValid;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "mrkd_statistic_impl.hpp"

#endif // __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP
