/**
 * @file fastmks.hpp
 * @author Ryan Curtin
 *
 * Definition of the FastMKS class, which is the fast max-kernel search.
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
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_HPP

#include <mlpack/core.hpp>
#include "ip_metric.hpp"
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>

namespace mlpack {
namespace fastmks {

template<typename KernelType>
class FastMKS
{
 public:
  FastMKS(const arma::mat& referenceSet,
          KernelType& kernel,
          bool single = false,
          bool naive = false,
          double expansionConstant = 2.0);

  FastMKS(const arma::mat& referenceSet,
          const arma::mat& querySet,
          KernelType& kernel,
          bool single = false,
          bool naive = false,
          double expansionConstant = 2.0);

  ~FastMKS();

  void Search(const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& products);

 private:
  const arma::mat& referenceSet;

  const arma::mat& querySet;

  tree::CoverTree<IPMetric<KernelType> >* referenceTree;

  tree::CoverTree<IPMetric<KernelType> >* queryTree;

  bool single;

  bool naive;

  IPMetric<KernelType> metric;

  // Utility function.  Copied too many times from too many places.
  void InsertNeighbor(arma::Mat<size_t>& indices,
                      arma::mat& products,
                      const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_impl.hpp"

#endif
