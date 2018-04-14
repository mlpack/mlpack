/**
 * @file output_size_visitor.hpp
 * @author Atharva Khandait
 *
 * This file provides an abstraction for the OutputSize() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_SIZE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_SIZE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * OutputWidthVisitor exposes the OutputSize() method of the given module.
 */
class OutputSizeVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the output size.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! Return 0 if the module doesn't implement the InputSize() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasInputSize<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T>::value, size_t>::type
  LayerOutputSize(T* layer) const;

  //! Return the output size if the module implements the InputSize()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasInputSize<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T>::value, size_t>::type
  LayerOutputSize(T* layer) const;

  //! Return the output size if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasInputSize<T, size_t&(T::*)()>::value &&
      HasModelCheck<T>::value, size_t>::type
  LayerOutputSize(T* layer) const;

  //! Return the output size if the module implement the Model() or
  //! InputSize() function.
  template<typename T>
  typename std::enable_if<
      HasInputSize<T, size_t&(T::*)()>::value &&
      HasModelCheck<T>::value, size_t>::type
  LayerOutputSize(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "output_size_visitor_impl.hpp"

#endif
