/**
 * @file methods/ann/visitor/loss_visitor.hpp
 * @author Atharva Khandait
 *
 * This file provides an abstraction for the Loss() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * LossVisitor exposes the Loss() method of the given module.
 */
class LossVisitor : public boost::static_visitor<double>
{
 public:
  //! Return the Loss.
  template<typename LayerType>
  double operator()(LayerType* layer) const;

  double operator()(MoreTypes layer) const;

 private:
  //! Return 0 if the module doesn't implement the Loss() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasLoss<T, double(T::*)()>::value &&
      !HasModelCheck<T>::value, double>::type
  LayerLoss(T* layer) const;

  //! Return the output height if the module implements the Loss() function.
  template<typename T>
  typename std::enable_if<
      HasLoss<T, double(T::*)()>::value &&
      !HasModelCheck<T>::value, double>::type
  LayerLoss(T* layer) const;

  //! Return the loss if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasLoss<T, double(T::*)()>::value &&
      HasModelCheck<T>::value, double>::type
  LayerLoss(T* layer) const;

  //! Return the loss if the module implements the Model() or loss() function.
  template<typename T>
  typename std::enable_if<
      HasLoss<T, double(T::*)()>::value &&
      HasModelCheck<T>::value, double>::type
  LayerLoss(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "loss_visitor_impl.hpp"

#endif
