/**
 * @file multiclass_classification_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the MulticlassClassificationLayer class, which implements a
 * multiclass classification layer that can be used as output layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_MULTICLASS_CLASSIFICATION_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_MULTICLASS_CLASSIFICATION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a multiclass classification layer that can be used as
 * output layer.
 *
 * A convenience typedef is given:
 *
 *  - ClassificationLayer
 *
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class MulticlassClassificationLayer
{
 public:
  /**
   * Create the MulticlassClassificationLayer object.
   */
  MulticlassClassificationLayer()
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
  void calculateError(const VecType& inputActivations,
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
  void outputClass(const VecType& inputActivations, VecType& output)
  {
    output = inputActivations;
  }
}; // class MulticlassClassificationLayer

//! Layer traits for the multiclass classification layer.
template <
    typename MatType,
    typename VecType
>
class LayerTraits<MulticlassClassificationLayer<MatType, VecType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
};

/***
 * Standard Input-Layer using the tanh activation function and the
 * Nguyen-Widrow method to initialize the weights.
 */
template <
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
using ClassificationLayer = MulticlassClassificationLayer<MatType, VecType>;

}; // namespace ann
}; // namespace mlpack


#endif
