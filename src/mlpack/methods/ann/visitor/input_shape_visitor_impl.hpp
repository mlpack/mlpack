/**
 * @file input_shape_visitor_impl.hpp
 * @author Nippun Sharma
 *
 * Implementation of the InputShape() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "input_shape_visitor.hpp"

namespace mlpack {
namespace ann {

//! InShapeVisitor visitor class.
template<typename LayerType>
inline std::size_t InShapeVisitor::operator()(LayerType* layer) const
{
  return LayerInputShape(layer);
}

inline std::size_t InShapeVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasInputShapeCheck<T>::value, std::size_t>::type
InShapeVisitor::LayerInputShape(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputShapeCheck<T>::value, std::size_t>::type
InShapeVisitor::LayerInputShape(T* layer) const
{
  return layer->InputShape();
}

} // namespace ann
} // namespace mlpack

#endif