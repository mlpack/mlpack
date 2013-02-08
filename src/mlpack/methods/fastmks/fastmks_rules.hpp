/**
 * @file fastmks_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for the single or dual tree traversal for fast max-kernel search.
 *
 * This file is part of MLPACK 1.0.4.
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
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>

namespace mlpack {
namespace fastmks {

template<typename MetricType>
class FastMKSRules
{
 public:
  FastMKSRules(const arma::mat& referenceSet,
               const arma::mat& querySet,
               arma::Mat<size_t>& indices,
               arma::mat& products);

  void BaseCase(const size_t queryIndex, const size_t referenceIndex);

  bool CanPrune(const size_t queryIndex,
                tree::CoverTree<MetricType>& referenceNode,
                const size_t parentIndex);

 private:
  const arma::mat& referenceSet;

  const arma::mat& querySet;

  arma::Mat<size_t>& indices;

  arma::mat& products;

  arma::vec queryKernels; // || q || for each q.

  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_rules_impl.hpp"

#endif
