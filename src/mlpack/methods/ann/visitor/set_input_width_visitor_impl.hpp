/**
 * @file set_input_width_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the InputWidth() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "set_input_width_visitor.hpp"

namespace mlpack {
namespace ann {

//! SetInputWidthVisitor visitor class.
inline SetInputWidthVisitor::SetInputWidthVisitor(const size_t inputWidth,
                                                  const bool reset) :
    inputWidth(inputWidth),
    reset(reset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool SetInputWidthVisitor::operator()(LayerType* layer) const
{
  return LayerInputWidth(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  if (layer->InputWidth() == 0 || reset)
  {
    layer->InputWidth() = inputWidth;
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
        layer->Model()[i]);
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  if (layer->InputWidth() == 0 || reset)
  {
    layer->InputWidth() = inputWidth;
  }

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
        layer->Model()[i]);
  }

  return true;
}

} // namespace ann
} // namespace mlpack

#endif
