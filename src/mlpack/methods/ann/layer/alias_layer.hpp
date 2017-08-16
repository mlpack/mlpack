/**
 * @file alias_layer.hpp
 * @author Ryan Curtin
 * @author Shangtong Zhang
 *
 * This is an alias layer for another layer so that parameters can be shared
 * between multiple networks.  However, it is not threadsafe---so you cannot
 * share parameters between networks in separate threads (although... it should
 * be pretty easy to adapt this class, you just need to add locks).
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include "layer_types.hpp"
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>
#include <mlpack/methods/ann/visitor/add_visitor.hpp>


namespace mlpack {
namespace ann {

class Alias
{
 public:
  /**
   * Construct the AliasLayer as an alias of the given layer.  When destructed,
   * this class will not destruct the aliased layer.
   *
   * @param layer Layer to be aliased.
   */
  template<typename LayerType>
  Alias(LayerType& layer) : layer(&layer) {};
  template<typename LayerType>
  Alias(LayerType* layer) : layer(layer) {};
  Alias(LayerTypes& layer) : layer(layer) {};

  /**
   * Reset the parameters of the layer.
   */
  void Reset()
  { boost::apply_visitor(ResetVisitor(), layer); }

  /**
   * Perform a forward pass of the aliased layer.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  /**
   * Perform a backwards pass of the aliased layer.
   */
  template<typename eT>
  void Backward(arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  /**
   * Calculate the gradient of the aliased layer using the delta and the input
   * activation.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  const arma::mat& OutputParameter() const { return weights; }
  arma::mat& OutputParameter() { return weights; }

  const arma::mat& Delta() const;
  arma::mat& Delta();

  template<typename LayerType>
  void Add(LayerType* newLayer)
  { boost::apply_visitor(AddVisitor(newLayer), layer); }

 private:
  LayerTypes layer;
  // Fake weights.
  arma::mat weights;
};

} // namespace ann
} // namespace mlpack

#include "alias_layer_impl.hpp"

#endif
