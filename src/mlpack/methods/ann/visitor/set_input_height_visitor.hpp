/**
 * @file set_input_height_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the InputHeight() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * SetInputHeightVisitor updates the input height parameter with the given input
 * height.
 */
class SetInputHeightVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the input height parameter with the given input height.
  SetInputHeightVisitor(const size_t inputHeight = 0, const bool reset = false);

  //! Update the input height parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

 private:
  //! The input height parameter.
  size_t inputHeight;

  //! If set reset the height parameter if already set.
  bool reset;

  //! Do nothing if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;

  //! Update the input height if the module implements the InputHeight() or
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerInputHeight(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "set_input_height_visitor_impl.hpp"

#endif
