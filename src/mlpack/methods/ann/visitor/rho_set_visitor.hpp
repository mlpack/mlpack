/**
 * @file rho_set_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Rho() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RHO_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_RHO_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * RhoSetVisitor updates the rho parameter with the given input sequence length.
 */
class RhoSetVisitor : public boost::static_visitor<bool>
{
 public:
  //! Update the rho parameter with the given sequence length.
  RhoSetVisitor(const size_t rho = 0);

  //! Update the rho parameter.
  template<typename LayerType>
  bool operator()(LayerType* layer) const;

 private:
  //! The sequence length.
  size_t rho;

  //! Do nothing if the module doesn't implement the Rho() or Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasRho<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerRho(T* layer) const;

  //! Update the rho if the module implements the Rho() function.
  template<typename T>
  typename std::enable_if<
      HasRho<T, size_t&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerRho(T* layer) const;

  //! Update the rho if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRho<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerRho(T* layer) const;

  //! Update the rho if the module implements the Rho() or Model() function.
  template<typename T>
  typename std::enable_if<
      HasRho<T, size_t&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
  LayerRho(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "rho_set_visitor_impl.hpp"

#endif