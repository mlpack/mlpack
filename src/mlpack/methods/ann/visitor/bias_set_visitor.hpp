/**
 * @file methods/ann/visitor/bias_set_visitor.hpp
 * @author Toshal Agrawal
 *
 * This file provides an abstraction for the Bias() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * BiasSetVisitor updates the module bias parameters given the parameters set.
 */
class BiasSetVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the bias parameters given the parameters' set and offset.
  BiasSetVisitor(arma::mat& weight, const size_t offset = 0);

  //! Update the parameters' set.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

  size_t operator()(MoreTypes layer) const;

 private:
  //! The parameters' set.
  arma::mat& weight;

  //! The parameters' offset.
  const size_t offset;

  //! Do not update the bias parameters if the module doesn't implement the
  //! Bias() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
      !HasModelCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;

  //! Update the bias parameters if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;

  //! Update the bias parameters if the module implements the Bias() function.
  template<typename T>
  typename std::enable_if<
      HasBiasCheck<T, arma::mat&(T::*)()>::value &&
      !HasModelCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;

  //! Update the bias parameters if the module implements the Model() and
  //! Bias() function.
  template<typename T>
  typename std::enable_if<
      HasBiasCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T>::value, size_t>::type
  LayerSize(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "bias_set_visitor_impl.hpp"

#endif
