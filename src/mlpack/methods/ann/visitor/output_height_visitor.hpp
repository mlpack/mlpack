/**
 * @file output_height_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the OutputHeight() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * OutputWidthVisitor exposes the OutputHeight() method of the given module.
 */
class OutputHeightVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output height.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputHeight() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the InputHeight()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;

  //! Return the output height if the module implement the Model() or
  //! InputHeight() function.
  template<typename T>
  typename std::enable_if<
      HasInputHeight<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerOutputHeight(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "output_height_visitor_impl.hpp"

#endif
