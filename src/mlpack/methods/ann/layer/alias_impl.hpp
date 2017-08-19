/**
 * @file alias_layer_impl.hpp
 * @author Ryan Curtin
 * @author Shangtong Zhang
 *
 * This is an alias layer for another layer so that parameters can be shared
 * between multiple networks.  However, it is not threadsafe---so you cannot
 * share parameters between networks in separate threads (although... it should
 * be pretty easy to adapt this class, you just need to add locks).
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_IMPL_HPP

#include "alias.hpp"

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/visitor/forward_visitor.hpp>
#include <mlpack/methods/ann/visitor/backward_visitor.hpp>
#include <mlpack/methods/ann/visitor/gradient_visitor.hpp>
#include <mlpack/methods/ann/visitor/delta_visitor.hpp>

namespace mlpack {
namespace ann {

/**
 * Perform a forward pass of the aliased layer.
 */
template<typename eT>
void Alias::Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Create a new visitor and call the layer's Forward() function.
//  arma::mat tmp;
//  boost::apply_visitor(ParametersVisitor(std::move(tmp)), layer);
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(output)),
      layer);
}

/**
 * Perform a backwards pass of the aliased layer.
 */
template<typename eT>
void Alias::Backward(arma::Mat<eT>&& input,
                          arma::Mat<eT>&& gy,
                          arma::Mat<eT>&& g)
{
  // Create a new visitor and call the layer's Backward() function.
  boost::apply_visitor(BackwardVisitor(std::move(input), std::move(gy),
      std::move(g)), layer);
}

/**
 * Calculate the gradient of the aliased layer using the delta and the input
 * activation.
 */
template<typename eT>
void Alias::Gradient(const arma::Mat<eT>&& input,
                          arma::Mat<eT>&& error,
                          arma::Mat<eT>&& gradient)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(error),
      std::move(gradient)), layer);
}

inline const arma::mat& Alias::Delta() const
{
  return boost::apply_visitor(DeltaVisitor(), layer);
}

inline arma::mat& Alias::Delta()
{
  return boost::apply_visitor(DeltaVisitor(), layer);
}

} // namespace ann
} // namespace mlpack

#endif
