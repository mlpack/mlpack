/**
 * @file methods/ann/layer/reparametrization.hpp
 * @author Atharva Khandait
 *
 * Definition of the Reparametrization layer class which samples from a gaussian
 * distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_HPP
#define MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"
// #include "../activation_functions/softplus_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Reparametrization layer class. This layer samples from the
 * given parameters of a normal distribution.
 *
 * This class also supports beta-VAE, a state-of-the-art framework for
 * automated discovery of interpretable factorised latent representations from
 * raw image data in a completely unsupervised manner.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{ICLR2017,
 *   title   = {beta-VAE: Learning basic visual concepts with a constrained
 *              variational framework},
 *   author  = {Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess,
 *              Xavier Glorot, Matthew Botvinick, Shakir Mohamed and
 *              Alexander Lerchner | Google DeepMind},
 *   journal = {2017 International Conference on Learning Representations(ICLR)},
 *   year    = {2017},
 *   url     = {https://deepmind.com/research/publications/beta-VAE-Learning-Basic-Visual-Concepts-with-a-Constrained-Variational-Framework}
 * }
 * @endcode
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class ReparametrizationType : public Layer<InputType, OutputType>
{
 public:
  //! Create the Reparametrization object.
  ReparametrizationType();

  /**
   * Create the Reparametrization layer object using the specified sample vector size.
   *
   * @param latentSize The number of output latent units.
   * @param stochastic Whether we want random sample or constant.
   * @param includeKl Whether we want to include KL loss in backward function.
   * @param beta The beta (hyper)parameter for beta-VAE mentioned above.
   */
  ReparametrizationType(const size_t latentSize,
                         const bool stochastic = true,
                         const bool includeKl = true,
                         const double beta = 1);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the output size.
  size_t const& OutputSize() const { return latentSize; }
  //! Modify the output size.
  size_t& OutputSize() { return latentSize; }

  //! Get the KL divergence with standard normal.
  double Loss()
  {
    if (!includeKl)
      return 0;

    return -0.5 * beta * arma::accu(2 * arma::log(stdDev) - arma::pow(stdDev, 2)
        - arma::pow(mean, 2) + 1) / mean.n_cols;
  }

  //! Get the value of the stochastic parameter.
  bool const& Stochastic() const { return stochastic; }

  //! Get the value of the includeKl parameter.
  bool const& IncludeKL() const { return includeKl; }

  //! Get the value of the beta hyperparameter.
  double const& Beta() const { return beta; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t latentSize;

  //! If false, sample will be constant.
  bool stochastic;

  //! If false, KL error will not be included in Backward function.
  bool includeKl;

  //! The beta hyperparameter for constrained variational frameworks.
  double beta;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored current gaussian sample.
  OutputType gaussianSample;

  //! Locally-stored current mean.
  OutputType mean;

  //! Locally-stored pre standard deviation.
  //! After softplus activation gives standard deviation.
  OutputType preStdDev;

  //! Locally-stored current standard deviation.
  OutputType stdDev;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class ReparametrizationType

// Standard Reparametrization layer.
typedef ReparametrizationType<arma::mat, arma::mat> Reparametrization;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reparametrization_impl.hpp"

#endif
