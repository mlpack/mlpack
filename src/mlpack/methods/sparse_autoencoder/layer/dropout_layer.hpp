/**
 * @file dropout_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the DropoutLayer class, which implements a regularizer that
 * randomly sets units to zero. Preventing units from co-adapting.
 */
#ifndef __MLPACK_METHODS_NN_LAYER_DROPOUT_LAYER_HPP
#define __MLPACK_METHODS_NN_LAYER_DROPOUT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_autoencoder/activation_functions/logistic_function.hpp>

namespace mlpack {
namespace nn /** Neural Network. */ {

/**
 * The dropout layer is a regularizer that randomly with probability ratio
 * sets input values to zero and scales the remaining elements by factor 1 /
 * (1 - ratio). If rescale is true the input is scaled with 1 / (1-p) when
 * deterministic is false. In the deterministic mode (during testing), the layer
 * just scales the output.
 *
 * Note: During training you should set deterministic to false and during
 * testing you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Hinton2012,
 *   author  = {Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky,
 *              Ilya Sutskever, Ruslan Salakhutdinov},
 *   title   = {Improving neural networks by preventing co-adaptation of feature
 *              detectors},
 *   journal = {CoRR},
 *   volume  = {abs/1207.0580},
 *   year    = {2012},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ActivationFunction = LogisticFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class DropoutLayer
{
 public:
  using ActivateFunction = ActivationFunction;

  /**
   * Create the BaseLayer object using the specified number of units.
   *
   * @param outSize The number of output units.
   */
  DropoutLayer(const double ratio = 0.5) :
      ratio(ratio),
      scale(1.0 / (1.0 - ratio))
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {    
    // Scale with input / (1 - ratio) and set values to zero with probability
    // ratio.
    scale = 1.0 / (1.0 - ratio);
    mask = arma::randu<arma::Mat<eT> >(input.n_rows, input.n_cols);
    mask.transform( [&](double val) { return (val > ratio); } );
    dropoutInput = input % mask * scale;
    ActivationFunction::fn(dropoutInput, output);
  } 

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& /* unused */,
                const DataType& gy,
                DataType& g)
  {
    g = gy % mask * scale;
  }  

  //! The probability of setting a value to zero.
  double Ratio() const {return ratio; }
  //! Modify the probability of setting a value to zero.
  double& Ratio() {return ratio; }  

 private: 
  //! Input after dropout
  InputDataType dropoutInput;
 
  //! Locally-stored mast object.
  OutputDataType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale; 
}; // class DropoutLayer

}; // namespace nn
}; // namespace mlpack

#endif
