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

/**
 * Implementation of the Reparametrization layer class. This layer samples from
 * the given parameters of a normal distribution.
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
 *   journal = {2017 International Conference on Learning Representations
 *                 (ICLR)},
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
  /**
   * Create the Reparametrization layer object.  Note that the inputs are
   * expected to be the parameters of the normal distribution; see the
   * documentation for Forward().
   *
   * @param stochastic Whether we want random sample or constant.
   * @param includeKl Whether we want to include KL loss in backward function.
   * @param beta The beta (hyper)parameter for beta-VAE mentioned above.
   */
  ReparametrizationType(const bool stochastic = true,
                        const bool includeKl = true,
                        const double beta = 1);

  /**
   * Clone the ReparametrizationType object. This handles polymorphism
   * correctly.
   */
  ReparametrizationType* Clone() const
  {
    return new ReparametrizationType(*this);
  }

  // Virtual destructor.
  virtual ~ReparametrizationType() { }

  //! Copy Constructor.
  ReparametrizationType(const ReparametrizationType& layer);

  //! Move Constructor.
  ReparametrizationType(ReparametrizationType&& layer);

  //! Copy assignment operator.
  ReparametrizationType& operator=(const ReparametrizationType& layer);

  //! Move assignment operator.
  ReparametrizationType& operator=(ReparametrizationType&& layer);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * Note that `input` is expected to be the parameters of the distribution.
   * The first `input.n_rows / 2` elements correspond to the
   * pre-standard-deviation values for each output element, and the second
   * `input.n_rows / 2` elements correspond to the means for each element.
   * Thus, the output size of the layer is the number of input elements divided
   * by 2.
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

  //! Get the KL divergence with standard normal.
  double Loss();

  //! Get the value of the stochastic parameter.
  bool Stochastic() const { return stochastic; }
  //! Modify the value of the stochastic parameter.
  bool& Stochastic() { return stochastic; }

  //! Get the value of the includeKl parameter.
  bool IncludeKL() const { return includeKl; }
  //! Modify the value of the includeKl parameter.
  bool& IncludeKL() { return includeKl; }

  //! Get the value of the beta hyperparameter.
  double Beta() const { return beta; }
  //! Modify the value of the beta hyperparameter.
  double& Beta() { return beta; }

  void ComputeOutputDimensions()
  {
    const size_t inputElem = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    if (inputElem % 2 != 0)
    {
      std::ostringstream oss;
      oss << "Reparametrization layer requires that the total number of input "
          << "elements is divisible by 2!  (Received input with " << inputElem
          << " total elements.)";
      throw std::invalid_argument(oss.str());
    }

    this->outputDimensions = std::vector<size_t>(
        this->inputDimensions.size(), 1);
    // This flattens the input, and removes half the elements.
    this->outputDimensions[0] = inputElem / 2;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! If false, sample will be constant.
  bool stochastic;

  //! If false, KL error will not be included in Backward function.
  bool includeKl;

  //! The beta hyperparameter for constrained variational frameworks.
  double beta;

  //! Locally-stored current gaussian sample.
  OutputType gaussianSample;

  //! Locally-stored current mean.
  OutputType mean;

  //! Locally-stored pre standard deviation.
  //! After softplus activation gives standard deviation.
  OutputType preStdDev;

  //! Locally-stored current standard deviation.
  OutputType stdDev;
}; // class ReparametrizationType

// Standard Reparametrization layer.
using Reparametrization = ReparametrizationType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "reparametrization_impl.hpp"

#endif
