/**
 * @file concat.hpp
 * @author Marcus Edel
 *
 * Definition of the Concat class, which acts as a concatenation contain.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"
#include "layer_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concat class. The Concat class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class Concat
{
 public:
  /**
   * Create the Concat object using the specified parameters.
   *
   * @param model Expose all network modules.
   * @param same Merge the error in the backward pass.
   */
  Concat(const bool model = true, const bool same = true) :
      model(model),
      same(same)
  {
    parameters.set_size(0, 0);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    size_t outSize = 0;

    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
          boost::apply_visitor(outputParameterVisitor, network[i]))),
          network[i]);

      if (boost::apply_visitor(
          outputParameterVisitor, network[i]).n_elem > outSize)
      {
        outSize = boost::apply_visitor(outputParameterVisitor,
            network[i]).n_elem;
      }
    }

    output = arma::zeros(outSize, network.size());
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t elements = boost::apply_visitor(outputParameterVisitor,
          network[i]).n_elem;

      if (elements < outSize)
      {
        output.submat(0, i, elements - 1, i) = arma::vectorise(
            boost::apply_visitor(outputParameterVisitor, network[i]));
      }
      else
      {
        output.col(i) = arma::vectorise(boost::apply_visitor(
          outputParameterVisitor, network[i]));
      }
    }
  }

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
  {
    size_t outSize = 0;
    size_t elements = 0;

    for (size_t i = 0, j = 0; i < network.size(); ++i, j += elements)
    {
      elements = boost::apply_visitor(outputParameterVisitor,
          network[i]).n_elem;

      arma::mat delta;
      if (gy.n_cols == 1)
      {
        delta = gy.submat(j, 0, j + elements - 1, 0);
      }
      else
      {
        delta = gy.submat(0, i, elements - 1, i);
      }

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, network[i])), std::move(delta), std::move(
          boost::apply_visitor(deltaVisitor, network[i]))), network[i]);

      if (boost::apply_visitor(deltaVisitor, network[i]).n_elem > outSize)
      {
        outSize = boost::apply_visitor(deltaVisitor, network[i]).n_elem;
      }

      if (same)
      {
        if (i == 0)
        {
          g = std::move(boost::apply_visitor(deltaVisitor, network[i]));
        }
        else
        {
          g += std::move(boost::apply_visitor(deltaVisitor, network[i]));
        }
      }
    }

    if (!same)
    {
      g = arma::zeros(outSize, network.size());
      for (size_t i = 0; i < network.size(); ++i)
      {
        size_t elements = boost::apply_visitor(deltaVisitor, network[i]).n_elem;
        if (elements < outSize)
        {
          g.submat(0, i, elements - 1, i) = arma::vectorise(
              boost::apply_visitor(deltaVisitor, network[i]));
        }
        else
        {
          g.col(i) = arma::vectorise(
              boost::apply_visitor(deltaVisitor, network[i]));
        }
      }
    }
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& /* gradient */)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, network[i])), std::move(error)), network[i]);
    }
  }

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<LayerTypes>& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameters; }

  arma::mat const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  arma::mat& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  arma::mat const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  arma::mat& OutputParameter() { return outputParameter; }

  //! Get the delta.e
  arma::mat const& Delta() const { return delta; }
  //! Modify the delta.
  arma::mat& Delta() { return delta; }

  //! Get the gradient.
  arma::mat const& Gradient() const { return gradient; }
  //! Modify the gradient.
  arma::mat& Gradient() { return gradient; }

 private:
  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! If true merge the error in the backward pass.
  bool same;

  //! Locally-stored network modules.
  std::vector<LayerTypes> network;

  //! Locally-stored model parameters.
  arma::mat parameters;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored empty list of modules.
  std::vector<LayerTypes> empty;

  //! Locally-stored delta object.
  arma::mat delta;

  //! Locally-stored input parameter object.
  arma::mat inputParameter;

  //! Locally-stored output parameter object.
  arma::mat outputParameter;

  //! Locally-stored gradient object.
  arma::mat gradient;
}; // class Concat


} // namespace ann
} // namespace mlpack

#endif
