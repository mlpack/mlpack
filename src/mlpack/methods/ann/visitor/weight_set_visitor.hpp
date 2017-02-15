/**
 * @file weight_set_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Weight() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * WeightSetVisitor update the module parameters given the parameters set.
 */
class WeightSetVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the parameters given the parameters set and offset.
  WeightSetVisitor(arma::mat&& weight, const size_t offset = 0);

  //! Update the parameters set.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! The parameters set.
  arma::mat&& weight;

  //! The parameters offset.
  const size_t offset;

  //! Do not update the parameters if the module doesn't implement the
  //! Parameters() or Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;

  //! Update the parameters if the module implements the Model() and
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerSize(T* layer, P&& input) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "weight_set_visitor_impl.hpp"

#endif
