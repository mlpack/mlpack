/**
 * @file binary_classification_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BinaryClassificationLayer class, which implements a
 * binary class classification layer that can be used as output layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a binary classification layer that can be used as
 * output layer.
 *
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class BinaryClassificationLayer
{
 public:
  /**
   * Create the BinaryClassificationLayer object.
   */
  BinaryClassificationLayer()
  {
    // Nothing to do here.
  }

  /*
   * Calculate the error using the specified input activation and the target.
   * The error is stored into the given error parameter.
   *
   * @param inputActivations Input data used for evaluating the network.
   * @param target Target data used for evaluating the network.
   * @param error The calculated error with respect to the input activation and
   * the given target.
   */
  void CalculateError(const VecType& inputActivations,
                      const VecType& target,
                      VecType& error)
  {
    error = inputActivations - target;
  }

  /*
   * Calculate the output class using the specified input activation.
   *
   * @param inputActivations Input data used to calculate the output class.
   * @param output Output class of the input activation.
   */
  void OutputClass(const VecType& inputActivations, VecType& output)
  {
    output = inputActivations;
    output.transform( [](double value) { return (value > 0.5 ? 1 : 0); } );
  }
}; // class BinaryClassificationLayer

//! Layer traits for the binary class classification layer.
template <typename MatType, typename VecType>
class LayerTraits<BinaryClassificationLayer<MatType, VecType> >
{
 public:
  static const bool IsBinary = true;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
