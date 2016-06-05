/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_IMPL_HPP

#include "lrsdp.hpp"

namespace mlpack {
namespace optimization {

template <typename SDPType>
LRSDP<SDPType>::LRSDP(const size_t numSparseConstraints,
                      const size_t numDenseConstraints,
                      const arma::mat& initialPoint) :
    function(numSparseConstraints, numDenseConstraints, initialPoint),
    augLag(function)
{ }

template <typename SDPType>
double LRSDP<SDPType>::Optimize(arma::mat& coordinates)
{
  augLag.Sigma() = 10;
  augLag.Optimize(coordinates, 1000);

  return augLag.Function().Evaluate(coordinates);
}

} // namespace optimization
} // namespace mlpack

#endif
