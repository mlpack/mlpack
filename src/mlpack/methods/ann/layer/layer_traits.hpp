/**
 * @file layer_traits.hpp
 * @author Marcus Edel
 *
 * This provides the LayerTraits class, a template class to get information
 * about various layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_TRAITS_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_TRAITS_HPP

#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace ann {

/**
 * This is a template class that can provide information about various layers.
 * By default, this class will provide the weakest possible assumptions on
 * layer, and each layer should override values as necessary.  If a layer
 * doesn't need to override a value, then there's no need to write a LayerTraits
 * specialization for that class.
 */
template<typename LayerType>
class LayerTraits
{
 public:
  /**
   * This is true if the layer is a binary layer.
   */
  static const bool IsBinary = false;

  /**
   * This is true if the layer is an output layer.
   */
  static const bool IsOutputLayer = false;

  /**
   * This is true if the layer is a bias layer.
   */
  static const bool IsBiasLayer = false;

  /*
   * This is true if the layer is a LSTM layer.
   **/
  static const bool IsLSTMLayer = false;

  /*
   * This is true if the layer is a connection layer.
   **/
  static const bool IsConnection = false;
};

// This gives us a HasGradientCheck<T, U> type (where U is a function pointer)
// we can use with SFINAE to catch when a type has a Gradient(...) function.
HAS_MEM_FUNC(Gradient, HasGradientCheck);

// This gives us a HasDeterministicCheck<T, U> type (where U is a function
// pointer) we can use with SFINAE to catch when a type has a Deterministic()
// function.
HAS_MEM_FUNC(Deterministic, HasDeterministicCheck);

// This gives us a HasRecurrentParameterCheck<T, U> type (where U is a function
// pointer) we can use with SFINAE to catch when a type has a
// RecurrentParameter() function.
HAS_MEM_FUNC(RecurrentParameter, HasRecurrentParameterCheck);

// This gives us a HasSeqLenCheck<T, U> type (where U is a function pointer) we
// can use with SFINAE to catch when a type has a SeqLen() function.
HAS_MEM_FUNC(SeqLen, HasSeqLenCheck);

// This gives us a HasWeightsCheck<T, U> type (where U is a function pointer) we
// can use with SFINAE to catch when a type has a Weights() function.
HAS_MEM_FUNC(Weights, HasWeightsCheck);

// This gives us a HasLocationCheck<T, U> type (where U is a function pointer)
// we can use with SFINAE to catch when a type has a Location() function.
HAS_MEM_FUNC(Location, HasLocationCheck);

// This gives us a HasRewardCheck<T, U> type (where U is a function pointer) we
// can use with SFINAE to catch when a type has a Reward() function.
HAS_MEM_FUNC(Reward, HasRewardCheck);

} // namespace ann
} // namespace mlpack

#endif
