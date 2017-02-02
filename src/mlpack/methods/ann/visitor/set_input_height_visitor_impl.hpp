/**
 * @file set_input_height_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the InputHeight() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "set_input_height_visitor.hpp"

namespace mlpack {
namespace ann {

//! SetInputHeightVisitor visitor class.
inline SetInputHeightVisitor::SetInputHeightVisitor(const size_t inputHeight,
                                                    const bool reset) :
    inputHeight(inputHeight),
    reset(reset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool SetInputHeightVisitor::operator()(LayerType* layer) const
{
  return LayerInputHeight(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  if (layer->InputHeight() == 0 || reset)
  {
    layer->InputHeight() = inputHeight;
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
        layer->Model()[i]);
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  if (layer->InputHeight() == 0  || reset)
  {
    layer->InputHeight() = inputHeight;
  }

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
        layer->Model()[i]);
  }

  return true;
}

} // namespace ann
} // namespace mlpack

#endif
