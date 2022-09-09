/**
 * @file methods/ann/ann.hpp
 * @author Ryan Curtin
 *
 * Convenience include for all aspects of the neural network framework in
 * mlpack.
 *
 * Note that serialization for neural networks is not enabled unless the
 * MLPACK_ENABLE_ANN_SERIALIZATION macro is defined!
 */
#ifndef MLPACK_METHODS_ANN_ANN_HPP
#define MLPACK_METHODS_ANN_ANN_HPP

#include "forward_decls.hpp"
#include "make_alias.hpp"

#include "activation_functions/activation_functions.hpp"
#include "augmented/augmented.hpp"
#include "convolution_rules/convolution_rules.hpp"
#include "dists/dists.hpp"
#include "init_rules/init_rules.hpp"
#include "layer/layer.hpp"
#include "loss_functions/loss_functions.hpp"
#include "regularizer/regularizer.hpp"

#include "ffn.hpp"
#include "rnn.hpp"

#endif
