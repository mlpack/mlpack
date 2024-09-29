/**
 * @file methods/ann/convolution_rules/border_modes.hpp
 * @author Marcus Edel
 *
 * This file provides the border modes that can be used to compute different
 * convolutions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP

namespace mlpack {

/*
 * The FullConvolution class represents the full two-dimensional convolution.
 */
class FullConvolution { /* Nothing to do here */ };

/*
 * The ValidConvolution represents only those parts of the convolution that are
 * computed without the zero-padded edges.
 */
class ValidConvolution { /* Nothing to do here */ };

} // namespace mlpack

#endif
