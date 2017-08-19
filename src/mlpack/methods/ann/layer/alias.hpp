/**
 * @file alias_layer.hpp
 * @author Ryan Curtin
 * @author Shangtong Zhang
 *
 * This is an alias layer for another layer so that parameters can be shared
 * between multiple networks.  However, it is not threadsafe---so you cannot
 * share parameters between networks in separate threads (although... it should
 * be pretty easy to adapt this class, you just need to add locks).
 *
 * You can only alias a layer which implements Parameters() and you
 * cannot alias a module.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_ALIAS_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/visitor/visitor.hpp>

namespace mlpack {
namespace ann {

//class WeightSizeVisitor;
//class ParametersVisitor;
//class ParametersSetVisitor;

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
  Alias(LayerType& layer) : layer(&layer) {
    parameter.set_size(boost::apply_visitor(WeightSizeVisitor(), this->layer), 1);
//    arma::mat tmp;
//    LayerTypes  l(this);
//    boost::apply_visitor(ParametersVisitor(std::move(tmp)), l);
  };
  template<typename LayerType>
  Alias(LayerType* layer) : layer(layer) {
    parameter.set_size(boost::apply_visitor(WeightSizeVisitor(), this->layer), 1);
  };
  Alias(LayerTypes& layer) : layer(layer) {
    parameter.set_size(boost::apply_visitor(WeightSizeVisitor(), this->layer), 1);
  };

  /**
   * Reset the parameters of the layer.
   */
  void Reset() {}
//  { boost::apply_visitor(ResetVisitor(), layer); }

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

  const arma::mat& OutputParameter() const {
    return boost::apply_visitor(OutputParameterVisitor(), layer);
  }
  arma::mat& OutputParameter() {
    return boost::apply_visitor(OutputParameterVisitor(), layer);
  }

  arma::mat& Parameters() { return parameter; }
  const arma::mat& Parameters() const { return parameter; }

  const arma::mat& Delta() const;
  arma::mat& Delta();

  template<typename LayerType>
  void Add(LayerType* newLayer)
  { boost::apply_visitor(AddVisitor(newLayer), layer); }

  size_t WeightSize() const
  { return boost::apply_visitor(WeightSizeVisitor(), layer); }

  size_t SetWeight(arma::mat&& weight, const size_t offset)
  {
    parameter = arma::mat(weight.memptr() + offset, parameter.n_rows, parameter.n_cols, false, false);
    return parameter.n_elem;
  }

  void SetParameters(arma::mat&& parameters)
  {
    parameter = static_cast<const arma::mat&>(parameters);
    boost::apply_visitor(ParametersSetVisitor(std::move(parameters)), layer);
  }

  void GetParameters(arma::mat&& parameters)
  {
    boost::apply_visitor(ParametersVisitor(std::move(parameters)), layer);
    parameter = static_cast<const arma::mat&>(parameters);
  }

 private:
  LayerTypes layer;
  arma::mat parameter;
};

} // namespace ann
} // namespace mlpack

#include "alias_impl.hpp"
#include <mlpack/methods/ann/visitor/visitor_impl.hpp>

#endif
