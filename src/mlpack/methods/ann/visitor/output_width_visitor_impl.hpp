/**
 * @file output_width_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the OutputWidth() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_WIDTH_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_WIDTH_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "output_width_visitor.hpp"

namespace mlpack {
namespace ann {

//! OutputWidthVisitor visitor class.
template<typename LayerType>
inline size_t OutputWidthVisitor::operator()(LayerType* layer) const
{
  return LayerOutputWidth(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  return layer->OutputWidth();
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputWidth = boost::apply_visitor(OutputWidthVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputWidth != 0)
    {
      return outputWidth;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  size_t outputWidth = layer->OutputWidth();

  if (outputWidth == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputWidth = boost::apply_visitor(OutputWidthVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputWidth != 0)
      {
        return outputWidth;
      }
    }
  }

  return outputWidth;
}

} // namespace ann
} // namespace mlpack

#endif
