/**
 * @file name_visitor_impl.hpp
 * @author Prince Gupta
 *
 * Implementation of the Name() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_NAME_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_NAME_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "name_visitor.hpp"

namespace mlpack {
namespace ann {

//! NameVisitor visitor class.
template<typename LayerType>
inline std::string NameVisitor::operator()(LayerType* layer) const
{
  return LayerName(layer);
}

inline std::string NameVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasName<T, std::string&(T::*)()>::value, std::string>::type
NameVisitor::LayerName(T* /* layer */) const
{
  return "unnamed";
}

template<typename T>
inline typename std::enable_if<
    HasName<T, std::string&(T::*)()>::value, std::string>::type
NameVisitor::LayerName(T* layer) const
{
  return layer->Name();
}

} // namespace ann
} // namespace mlpack

#endif
