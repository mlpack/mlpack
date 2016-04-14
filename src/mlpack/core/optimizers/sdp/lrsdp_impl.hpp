/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
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
