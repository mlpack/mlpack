/**
 * @file in_size_visitor_impl.hpp
 * @author Khizir Siddiqui
 *
 * Implementation of the InputSize() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_IN_SIZE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_IN_SIZE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "in_size_visitor.hpp"

namespace mlpack {
namespace ann {

//! WeightSizeVisitor visitor class.
template<typename LayerType>
inline std::size_t InSizeVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer);
}

inline std::size_t InSizeVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasInputSizeCheck<T>::value, std::size_t>::type
InSizeVisitor::LayerSize(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputSizeCheck<T>::value, std::size_t>::type
InSizeVisitor::LayerSize(T* layer) const
{
  return layer->InputSize();
}

} // namespace ann
} // namespace mlpack

#endif
