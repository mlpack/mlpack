/**
 * @file output_size_visitor_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the OutputSize() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_SIZE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_SIZE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "output_size_visitor.hpp"

namespace mlpack {
namespace ann {

//! OutputSizeVisitor visitor class.
template<typename LayerType>
inline size_t OutputSizeVisitor::operator()(LayerType* layer) const
{
  return LayerOutputSize(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputSize<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T>::value, size_t>::type
OutputSizeVisitor::LayerOutputSize(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputSize<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T>::value, size_t>::type
OutputSizeVisitor::LayerOutputSize(T* layer) const
{
  return layer->OutputSize();
}

template<typename T>
inline typename std::enable_if<
    !HasInputSize<T, size_t&(T::*)()>::value &&
    HasModelCheck<T>::value, size_t>::type
OutputSizeVisitor::LayerOutputSize(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outSize = boost::apply_visitor(OutputSizeVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outSize != 0)
    {
      return outSize;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputSize<T, size_t&(T::*)()>::value &&
    HasModelCheck<T>::value, size_t>::type
OutputSizeVisitor::LayerOutputSize(T* layer) const
{
  size_t outSize = layer->OutputSize();

  if (outSize == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outSize = boost::apply_visitor(OutputSizeVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outSize != 0)
      {
        return outSize;
      }
    }
  }

  return outSize;
}

} // namespace ann
} // namespace mlpack

#endif
