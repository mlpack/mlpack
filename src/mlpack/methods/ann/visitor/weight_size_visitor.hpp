/**
 * @file weight_size_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the WeightSize() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * WeightSizeVisitor returns the number of weights of the given module.
 */
class WeightSizeVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Return the number of weights.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! If the module doesn't implement the Parameters() or Model() function
  //! return 0.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the number of parameters if the module implements the Model()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the number of parameters if the module implements the Parameters()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;

  //! Return the accumulated number of parameters if the module implements the
  //! Parameters() and Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P& output) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "weight_size_visitor_impl.hpp"

#endif
