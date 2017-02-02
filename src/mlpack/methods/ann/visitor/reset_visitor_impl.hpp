/**
 * @file reset_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Reset() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "reset_visitor.hpp"

namespace mlpack {
namespace ann {

//! ResetVisitor visitor class.
template<typename LayerType>
inline void ResetVisitor::operator()(LayerType* layer) const
{
  ResetParameter(layer);
}

template<typename T>
inline typename std::enable_if<
    HasResetCheck<T, void(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  layer->Reset();
}

template<typename T>
inline typename std::enable_if<
    !HasResetCheck<T, void(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasResetCheck<T, void(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
  }

  layer->Reset();
}

template<typename T>
inline typename std::enable_if<
    !HasResetCheck<T, void(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
