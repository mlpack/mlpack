/**
 * @file set_input_size_visitor_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the InputSize() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_SIZE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_SIZE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "set_input_size_visitor.hpp"

namespace mlpack {
namespace ann {

//! SetInputSizeVisitor visitor class.
inline SetInputSizeVisitor::SetInputSizeVisitor(const size_t inSize,
                                                const bool reset) :
    inSize(inSize),
    reset(reset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool SetInputSizeVisitor::operator()(LayerType* layer) const
{
  return LayerInputSize(layer);
}

inline bool SetInputSizeVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    (!HasInputSize<T, size_t&(T::*)()>::value ||
     !HasResetCheck<T, void(T::*)()>::value) &&
    !HasModelCheck<T>::value, bool>::type
SetInputSizeVisitor::LayerInputSize(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasInputSize<T, size_t&(T::*)()>::value &&
    HasResetCheck<T, void(T::*)()>::value &&
    !HasModelCheck<T>::value, bool>::type
SetInputSizeVisitor::LayerInputSize(T* layer) const
{
  if (layer->InputSize() == 0 || reset)
  {
    layer->InputSize() = inSize;
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    (!HasInputSize<T, size_t&(T::*)()>::value ||
     !HasResetCheck<T, void(T::*)()>::value) &&
    HasModelCheck<T>::value, bool>::type
SetInputSizeVisitor::LayerInputSize(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputSizeVisitor(inSize, reset),
        layer->Model()[i]);
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasInputSize<T, size_t&(T::*)()>::value &&
    HasResetCheck<T, void(T::*)()>::value &&
    HasModelCheck<T>::value, bool>::type
SetInputSizeVisitor::LayerInputSize(T* layer) const
{
  if (layer->InputSize() == 0  || reset)
  {
    layer->InputSize() = inSize;
    layer->Reset();
  }

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputSizeVisitor(inSize, reset),
        layer->Model()[i]);
  }

  return true;
}

} // namespace ann
} // namespace mlpack

#endif
