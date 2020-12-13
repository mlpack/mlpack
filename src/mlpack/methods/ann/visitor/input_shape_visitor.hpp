/**
 * @file input_shape_visitor.hpp
 * @author Khizir Siddiqui
 * @author Nippun Sharma
 *
 * This file provides an abstraction for the InputShape() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * InShapeVisitor returns the input shape a Layer expects.
 */
class InShapeVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the input shape of layer.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

  size_t operator()(MoreTypes layer) const;

 private:
  //! If the module doesn't implement the InputShape() function return 0.
  template<typename T>
  typename std::enable_if<
      !HasInputShapeCheck<T>::value, size_t>::type
  LayerInputShape(T* layer) const;

  //! If the module implements the InputShape() function returns the input shape.
  template<typename T>
  typename std::enable_if<
      HasInputShapeCheck<T>::value, size_t>::type
  LayerInputShape(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "input_shape_visitor_impl.hpp"

#endif
