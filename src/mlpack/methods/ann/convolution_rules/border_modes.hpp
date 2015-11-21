/**
 * @file border_modes.hpp
 * @author Marcus Edel
 *
 * This file provides the border modes that can be used to compute different
 * convolutions.
 */
#ifndef __MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP
#define __MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP

namespace mlpack {
namespace ann {

/*
 * The FullConvolution class represents the full two-dimensional convolution.
 */
class FullConvolution { /* Nothing to do here */ };

/*
 * The ValidConvolution represents only those parts of the convolution that are
 * computed without the zero-padded edges.
 */
class ValidConvolution { /* Nothing to do here */ };

} // namespace ann
} // namespace mlpack

#endif
