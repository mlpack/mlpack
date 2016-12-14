/**
 * @file dropout.hpp
 * @author Marcus Edel
 *
 * Definition of the Dropout class, which implements a regularizer that
 * randomly sets units to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class Dropout
{
 public:
  /**
   * Create the Dropout object using the specified ratio and rescale
   * parameter.
   *
   * @param ratio The probability of setting a value to zero.
   * @param rescale If true the input is rescaled when deterministic is False.
   */
<<<<<<< HEAD
<<<<<<< HEAD
  Dropout(const double ratio = 0.5, const bool rescale = true);
=======
  Dropout(const double ratio = 0.5,
          const bool rescale = true) :
      ratio(ratio),
      scale(1.0 / (1.0 - ratio)),
      rescale(rescale)
  {
    // Nothing to do here.
  }
>>>>>>> Refactor ann layer.
=======
  Dropout(const double ratio = 0.5, const bool rescale = true);
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
<<<<<<< HEAD
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      if (!rescale)
      {
        output = input;
      }
      else
      {
        output = input * scale;
      }
    }
    else
    {
      // Scale with input / (1 - ratio) and set values to zero with probability
      // ratio.
      mask = arma::randu<arma::Mat<eT> >(input.n_rows, input.n_cols);
      mask.transform( [&](double val) { return (val > ratio); } );
      output = input % mask * scale;
    }
  }
>>>>>>> Refactor ann layer.
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    g = gy % mask * scale;
  }
>>>>>>> Refactor ann layer.
=======
                arma::Mat<eT>&& g);
>>>>>>> Split layer modules into definition and implementation.

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the detla.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  //! The value of the rescale parameter.
  bool Rescale() const {return rescale; }
  //! Modify the value of the rescale parameter.
  bool& Rescale() {return rescale; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
<<<<<<< HEAD
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(ratio, "ratio");
    ar & data::CreateNVP(rescale, "rescale");
  }
>>>>>>> Refactor ann layer.
=======
  void Serialize(Archive& ar, const unsigned int /* version */);
>>>>>>> Split layer modules into definition and implementation.

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mast object.
  OutputDataType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! If true the input is rescaled when deterministic is False.
  bool rescale;
}; // class Dropout

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
<<<<<<< HEAD
// Include implementation.
#include "dropout_impl.hpp"

=======
>>>>>>> Refactor ann layer.
=======
// Include implementation.
#include "dropout_impl.hpp"

>>>>>>> Split layer modules into definition and implementation.
#endif
