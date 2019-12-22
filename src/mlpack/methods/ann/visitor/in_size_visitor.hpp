/**
 * @file in_size_visitor.hpp
 * @author Khizir Siddiqui
 *
 * This file provides an abstraction for the InSize() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_IN_SIZE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_IN_SIZE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * InSizeVisitor returns the size of Input in Layer.
 */
class InSizeVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the input size of layer.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

  size_t operator()(MoreTypes layer) const;

 private:
  //! If the module doesn't implement the InSize() function
  //! return 0.
  template<typename T>
  typename std::enable_if<
      !HasInputSizeCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;

  //! If the module implements the InSize() function
  //! returns the inSize.
  template<typename T>
  typename std::enable_if<
      HasInputSizeCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "in_size_visitor_impl.hpp"

#endif
