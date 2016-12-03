/**
 * @file dropconnect.hpp
 * @author Palash Ahuja
 * @author Marcus Edel
 *
 * Definition of the DropConnect class, which implements a regularizer
 * that randomly sets connections to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP

#include <mlpack/core.hpp>

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "linear.hpp"
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The DropConnect layer is a regularizer that randomly with probability
 * ratio sets the connection values to zero and scales the remaining
 * elements by factor 1 /(1 - ratio). The output is scaled with 1 / (1 - p)
 * when deterministic is false. In the deterministic mode(during testing),
 * the layer just computes the output. The output is computed according
 * to the input layer. If no input layer is given, it will take a linear layer
 * as default.
 *
 * Note:
 * During training you should set deterministic to false and during testing
 * you should set deterministic to true.
 *
 *  For more information, see the following.
 *
 * @code
 * @inproceedings{WanICML2013,
 *   title={Regularization of Neural Networks using DropConnect},
 *   booktitle = {Proceedings of the 30th International Conference on Machine
 *                Learning(ICML - 13)},
 *   author = {Li Wan and Matthew Zeiler and Sixin Zhang and Yann L. Cun and
 *             Rob Fergus},
 *   year = {2013}
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class DropConnect
{
 public:
  //! Create the DropConnect object.
  DropConnect()
  {
    /* Nothing to do here. */
  }

  /**
   * Creates the DropConnect Layer as a Linear Object that takes input size,
   * output size and ratio as parameter.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param ratio The probability of setting a value to zero.
   */
  DropConnect(const size_t inSize,
              const size_t outSize,
              const double ratio = 0.5) :
      ratio(ratio),
      scale(1.0 / (1 - ratio)),
      baseLayer(new Linear<InputDataType, OutputDataType>(inSize, outSize))
  {
    network.push_back(baseLayer);
  }

  ~DropConnect()
  {
    boost::apply_visitor(DeleteVisitor(), baseLayer);
  }

  /**
  * Ordinary feed forward pass of the DropConnect layer.
  *
  * @param input Input data used for evaluating the specified function.
  * @param output Resulting output activation.
  */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    // The DropConnect mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      boost::apply_visitor(
        ForwardVisitor(
          std::move(input),
          std::move(output)
        ),
        baseLayer);
    }
    else
    {
      // Save weights for denoising.
      boost::apply_visitor(ParametersVisitor(std::move(denoise)), baseLayer);

      // Scale with input / (1 - ratio) and set values to zero with
      // probability ratio.
      mask = arma::randu<arma::Mat<eT> >(denoise.n_rows, denoise.n_cols);
      mask.transform([&](double val) { return (val > ratio); });

      boost::apply_visitor(ParametersSetVisitor(std::move(denoise % mask)),
          baseLayer);

      boost::apply_visitor(
        ForwardVisitor(
          std::move(input),
          std::move(output)
        ),
        baseLayer);

      output = output * scale;
    }
  }

  /**
   * Ordinary feed backward pass of the DropConnect layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
  {
    boost::apply_visitor(
      BackwardVisitor(
          std::move(input),
          std::move(gy),
          std::move(g)
      ),
      baseLayer);
  }

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input.
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& /* gradient */)
  {
    boost::apply_visitor(GradientVisitor(std::move(input), std::move(error)),
        baseLayer);

    // Denoise the weights.
    boost::apply_visitor(ParametersSetVisitor(std::move(denoise)), baseLayer);
  }

  //! Get the model modules.
  std::vector<LayerTypes>& Model() { return network; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return parameters; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return parameters; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }

  //! Modify the value of the deterministic parameter.
  bool &Deterministic() { return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

private:
  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! Locally-stored weight object.
  OutputDataType parameters;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mask object.
  OutputDataType mask;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Denoise mask for the weights.
  OutputDataType denoise;

  //! Locally-stored layer module.
  LayerTypes baseLayer;

  //! Locally-stored network modules.
  std::vector<LayerTypes> network;
}; // class DropConnect.

}  // namespace ann
}  // namespace mlpack

#endif
