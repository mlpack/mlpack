#ifndef __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
#define __MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nn {


/**
 * Maximize the hidden units of the parameters
 * @param params The parameters want to maximize
 * @param output Parameters after maximize
 */
void MaximalInputs(arma::mat const &parameters, arma::mat &output);

} // namespace nn
} // namespace mlpack

#endif
