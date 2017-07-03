/**
 * @file ssRBM.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RBM_SPIKE_SLAB_SPIKE_SLAB_LAYER_IMPL_HPP
#define MLPACK_METHODS_RBM_SPIKE_SLAB_SPIKE_SLAB_LAYER_IMPL_HPP
// In case it hasn't yet been included.
#include "spike_slab_layer.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace rbm{
  template<typename VisibleLayer, typename HiddenLayer>
  class ssRBM
  {
  public:
    ssRBM(VisibleLayer visible, HiddenLayer hidden, const size_t numSteps, bool persisitence,)
    double FreeEnergy(arma::mat&& input);


    
    
  };
} // namespace rbm
} // namespace mlpack