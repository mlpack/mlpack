/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LRSDP_IMPL_HPP
#define ENSMALLEN_SDP_LRSDP_IMPL_HPP

#include "lrsdp.hpp"

namespace ens {

template <typename SDPType>
LRSDP<SDPType>::LRSDP(const size_t numSparseConstraints,
                      const size_t numDenseConstraints,
                      const arma::mat& initialPoint,
                      const size_t maxIterations) :
    function(numSparseConstraints, numDenseConstraints, initialPoint),
    maxIterations(maxIterations)
{ }

template <typename SDPType>
double LRSDP<SDPType>::Optimize(arma::mat& coordinates)
{
  augLag.Sigma() = 10;
  augLag.Optimize(function, coordinates, maxIterations);

  return function.Evaluate(coordinates);
}

} // namespace ens

#endif
