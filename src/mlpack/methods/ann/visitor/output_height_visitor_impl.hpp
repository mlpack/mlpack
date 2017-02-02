/**
 * @file output_height_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the OutputHeight() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "output_height_visitor.hpp"

namespace mlpack {
namespace ann {

//! OutputHeightVisitor visitor class.
template<typename LayerType>
inline size_t OutputHeightVisitor::operator()(LayerType* layer) const
{
  return LayerOutputHeight(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  return layer->OutputHeight();
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputHeight = boost::apply_visitor(OutputHeightVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputHeight != 0)
    {
      return outputHeight;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  size_t outputHeight = layer->OutputHeight();

  if (outputHeight == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputHeight = boost::apply_visitor(OutputHeightVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputHeight != 0)
      {
        return outputHeight;
      }
    }
  }

  return outputHeight;
}

} // namespace ann
} // namespace mlpack

#endif
