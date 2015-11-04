#ifndef __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
#define __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nn {

/**
 * Maximize the hidden units of the parameters, details are located at
 * http://deeplearning.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder.
 * This function is based on the implementation(display_network.m) from the exercise of UFLDL.
 * http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
 * @param params The parameters want to maximize
 * @param output Parameters after maximize
 * @return Suggestion rows and cols for the function "ColumnsToBlocks"
 * @pre 1 : The layout of the parameters should be same as following
 * //          vSize   1
 * //       |        |  |
 * //  hSize|   w1   |b1|
 * //       |________|__|
 * //       |        |  |
 * //  hSize|   w2'  |  |
 * //       |________|__|
 * //      1|   b2'  |  |
 *
 * 2 : Square root of vSize must be interger and bigger than zero
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
 * const auto Size = mlpack::nn::MaximalInputs(encoder2.Parameters(), maximalInput);
 *
 * arma::mat outputs;
 * const bool scale = true;
 * //put the maximalInput of each col into rows * cols(Size.first * Size.second) blocks,
 * //rows * cols must same as maximalInput.n_cols.If you are training a bunch of images,
 * //this function could help you visualize your trained results
 * mlpack::nn::ColumnsToBlocks(maximalInput, outputs, Size.first, Size.second, scale);
 *
 * @endcode
 */
std::pair<arma::uword, arma::uword> MaximalInputs(const arma::mat &parameters, arma::mat &output);

/**
 * Transform the output of "MaximalInputs" to blocks, if your training samples are images,
 * this function could help you visualize your training results
 * @param maximalInputs Parameters after maximize by "MaximalInputs", each col assiociate to one sample
 * @param output Maximal inputs regrouped to blocks
 * @param rows number of blocks per cols
 * @param cols number of blocks per rows
 * @param scale False, the output will not be scaled and vice versa
 * @param minRange minimum range of the output
 * @param maxRange maximum range of the output
 * @code
 * arma::mat maximalInput; //store the features learned by sparse autoencoder
 * const auto Size = mlpack::nn::MaximalInputs(encoder2.Parameters(), maximalInput);
 *
 * arma::mat outputs;
 * const bool scale = true;
 * //put the maximalInput of each col into rows * cols(Size.first * Size.second) blocks,
 * //rows * cols must same as maximalInput.n_cols.If you are training a bunch of images,
 * //this function could help you visualize your trained results
 * mlpack::nn::ColumnsToBlocks(maximalInput, outputs, Size.first, Size.second, scale);
 * @endcode
 */
void ColumnsToBlocks(const arma::mat &maximalInputs,
                     arma::mat &outputs, arma::uword rows, arma::uword cols,
                     bool scale = false,
                     double minRange = 0, double maxRange = 255);

} // namespace nn
} // namespace mlpack

#endif
