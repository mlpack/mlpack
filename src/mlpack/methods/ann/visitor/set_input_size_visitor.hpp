/**
 * @file set_input_size_visitor.hpp
 * @author Atharva Khandait
 *
 * This file provides an abstraction for the InputSize() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_SIZE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_SIZE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * SetInputSizeVisitor updates the input size parameter with the given input
 * size.
 */
class SetInputSizeVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the input size parameter with the given input size.
  SetInputSizeVisitor(const size_t inSize = 0, const bool reset = false);

  //! Update the input size parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

  bool operator()(MoreTypes layer) const;

 private:
  //! The input size parameter.
  size_t inSize;

  //! If set reset the size parameter if already set.
  bool reset;

  //! Do nothing if the module doesn't implement the InputSize() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputSize<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T>::value, bool>::type
  LayerInputSize(T* layer) const;

  //! Update the input size if the module implements the InputSize() function.
  template<typename T>
  typename std::enable_if<
      HasInputSize<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T>::value, bool>::type
  LayerInputSize(T* layer) const;

  //! Update the input size if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputSize<T, size_t&(T::*)()>::value &&
      HasModelCheck<T>::value, bool>::type
  LayerInputSize(T* layer) const;

  //! Update the input size if the module implements the InputSize() or
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      HasInputSize<T, size_t&(T::*)()>::value &&
      HasModelCheck<T>::value, bool>::type
  LayerInputSize(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "set_input_size_visitor_impl.hpp"

#endif
