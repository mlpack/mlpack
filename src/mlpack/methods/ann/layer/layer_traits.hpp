/**
 * @file layer_traits.hpp
 * @author Mark Fischinger 
 *
 * Definition of layer traits for type conversion in neural network layers.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_TRAITS_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_TRAITS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

/**
 * LayerTraits provides type information and conversion utilities for neural network layers.
 */
template<typename LayerType>
struct LayerTraits
{
  // The matrix type used by this layer.
  using MatType = typename LayerType::MatType;

  /**
   * Convert a layer to use a different matrix type.
   *
   * @param layer The layer to convert.
   * @return A new layer with the converted matrix type.
   */
  template<typename TargetMatType>
  static auto Convert(const LayerType& layer)
      -> typename LayerType::template LayerType<TargetMatType>
  {
    return typename LayerType::template LayerType<TargetMatType>(layer);
  }
};

} // namespace ann
} // namespace mlpack

#endif // MLPACK_METHODS_ANN_LAYER_LAYER_TRAITS_HPP