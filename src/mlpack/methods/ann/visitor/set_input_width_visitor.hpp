/**
 * @file set_input_width_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the InputWidth() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * SetInputWidthVisitor updates the input width parameter with the given input
 * width.
 */
class SetInputWidthVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the input width parameter with the given input width.
  SetInputWidthVisitor(const size_t inputWidth = 0, const bool reset = false);

  //! Update the input width parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

 private:
  //! The input width parameter.
  size_t inputWidth;

  //! If set reset the height parameter if already set.
  bool reset;

  //! Do nothing if the module doesn't implement the InputWidth() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the InputWidth() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;

  //! Update the input width if the module implements the InputWidth() or
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      HasInputWidth<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputWidth(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "set_input_width_visitor_impl.hpp"

#endif
