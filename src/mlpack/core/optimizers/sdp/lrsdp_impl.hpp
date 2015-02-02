/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_IMPL_HPP

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

// Convert the object to a string.
template <typename SDPType>
std::string LRSDP<SDPType>::ToString() const
{
  std::ostringstream convert;
  convert << "LRSDP [" << this << "]" << std::endl;
  convert << "  Optimizer: " << util::Indent(augLag.ToString(), 1) << std::endl;
  return convert.str();
}

}; // namespace optimization
}; // namespace mlpack

#endif
