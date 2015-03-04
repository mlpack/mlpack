/**
 * @file one_hot_layer.hpp
 * @author Shangtong Zhang
 *
 * Definition of the OneHotLayer class, which implements a standard network
 * layer.
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_ONE_HOT_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_ONE_HOT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a one hot classification layer that can be used as
 * output layer.
 *
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class OneHotLayer
{
 public:
  /**
   * Create the OneHotLayer object.
   */
  OneHotLayer()
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
                      VecType& error) {
    error = inputActivations - target;
  }

  /*
   * Calculate the output class using the specified input activation.
   *
   * @param inputActivations Input data used to calculate the output class.
   * @param output Output class of the input activation.
   */
  void outputClass(const VecType& inputActivations, VecType& output) {
    output = arma::zeros<VecType>(inputActivations.n_elem);
    arma::uword maxIndex;
    inputActivations.max(maxIndex);
    output(maxIndex) = 1;
  }
}; // class OneHotLayer

//! Layer traits for the one-hot class classification layer.
template <
    typename MatType,
    typename VecType
>
class LayerTraits<OneHotLayer<MatType, VecType> >
{
 public:
  static const bool IsBinary = true;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
};

}; // namespace ann
}; // namespace mlpack


#endif
