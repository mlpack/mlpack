/**
 * @file maximal_inputs.hpp
 * @author Tham Ngap Wei
 *
 * A function to find the maximal inputs of an autoencoder.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
#define MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nn {

/**
 * Given a parameters matrix from an autoencoder, maximize the hidden units of
 * the parameters, storing the maximal inputs in the given output matrix.
 * Details can be found on the 'Visualizing a Trained Autoencoder' page of the
 * Stanford UFLDL tutorial:
 *
 * http://deeplearning.stanford.edu/wiki/index.php/Main_Page
 *
 * This function is based on the implementation (display_network.m) from the
 * "Exercise: Sparse Autoencoder" page of the UFLDL tutorial:
 *
 * http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
 *
 * Example usage of this function can be seen below.  Note that this function
 * can work with the ColumnsToBlocks class in order to reshape the maximal
 * inputs for visualization, as in the UFLDL tutorial.  The code below
 * demonstrates this.
 *
 * @code
 * arma::mat data; // Data matrix.
 * const size_t vSize = 64; // Size of visible layer, depends on the data.
 * const size_t hSize = 25; // Size of hidden layer, depends on requirements.
 *
 * const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
 * const size_t numIterations = 100; // Maximum number of iterations.
 *
 * // Use an instantiated optimizer for the training.
 * SparseAutoencoder<L_BFGS> encoder(data, vSize, hSize);
 *
 * arma::mat maximalInput; // Store the features learned by sparse autoencoder
 * mlpack::nn::MaximalInputs(encoder.Parameters(), maximalInput);
 *
 * arma::mat outputs;
 * const bool scale = true;
 *
 * ColumnsToBlocks ctb(5,5);
 * arma::mat output;
 * ctb.Transform(maximalInput, output);
 * // Save the output as PGM, for visualization.
 * output.save(fileName, arma::pgm_binary);
 * @endcode
 *
 * @pre Layout of parameters
 *
 * The layout of the parameters matrix should be same as following
 * @code
 * //          vSize   1
 * //       |        |  |
 * //  hSize|   w1   |b1|
 * //       |________|__|
 * //       |        |  |
 * //  hSize|   w2'  |  |
 * //       |________|__|
 * //      1|   b2'  |  |
 * @endcode
 *
 * Also, the square root of vSize must be an integer (i.e. vSize must be a
 * perfect square).
 *
 * @param parameters The parameters of the autoencoder.
 * @param output Matrix to store the maximal inputs in.
 */
void MaximalInputs(const arma::mat& parameters, arma::mat& output);

/**
 * Normalize each column of the input matrix by its maximum value, if that
 * maximum value is not zero.
 *
 * @param input The input data to normalize.
 * @param output A matrix to store the input data in after normalization.
 */
void NormalizeColByMax(const arma::mat& input, arma::mat& output);

} // namespace nn
} // namespace mlpack

#endif
