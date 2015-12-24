/**
 * @file border_modes.hpp
 * @author Marcus Edel
 *
 * This file provides the border modes that can be used to compute different
 * convolutions.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
