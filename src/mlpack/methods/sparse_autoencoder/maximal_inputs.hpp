#ifndef __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
#define __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nn {


/**
 * Maximize the hidden units of the parameters
 * @param params The parameters want to maximize
 * @param output Parameters after maximize
 * @code
 * arma::mat data; // Data matrix.
 * const size_t vSize = 64; // Size of visible layer, depends on the data.
 * const size_t hSize = 25; // Size of hidden layer, depends on requirements.
 *
 *
 * const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
 * const size_t numIterations = 100; // Maximum number of iterations.
 *
 * // Use an instantiated optimizer for the training.
 * SparseAutoencoderFunction saf(data, vSize, hSize);
 * L_BFGS<SparseAutoencoderFunction> optimizer(saf, numBasis, numIterations);
 * SparseAutoencoder<L_BFGS> encoder2(optimizer);
 *
 * arma::mat maximalInput; //store the features learned by sparse autoencoder
 * mlpack::nn::MaximalInputs(encoder2.Parameters(), maximalInput);
 * maximalInput.save("trained.pgm", arma::pgm_binary);
 * @endcode
 */
void MaximalInputs(arma::mat const &parameters, arma::mat &output);

} // namespace nn
} // namespace mlpack

#endif
