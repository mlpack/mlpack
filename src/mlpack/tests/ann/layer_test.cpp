/**
 * @file layer_test.cpp
 * @author Ryan Curtin
 *
 * This file includes all tests in ann/layer/, which are split up by layer for
 * organization.  However, compiling each test individually results in a huge
 * amount of compilation overhead; including them all here into one file reduces
 * compilation time and memory usage.
 *
 * It's possible that this could be avoided by smart use of extern template
 * instantiations.
 */

#include "layer/adaptive_max_pooling.cpp"
#include "layer/adaptive_mean_pooling.cpp"
#include "layer/alpha_dropout.cpp"
#include "layer/convolution.cpp"
#include "layer/dropout.cpp"
#include "layer/linear3d.cpp"
#include "layer/linear_no_bias.cpp"
#include "layer/log_softmax.cpp"
#include "layer/max_pooling.cpp"
#include "layer/mean_pooling.cpp"
#include "layer/padding.cpp"
#include "layer/softmax.cpp"
