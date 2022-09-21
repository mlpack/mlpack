/**
 * @file forward_decls.hpp
 * @author Ryan Curtin
 *
 * Forward declarations of network types.  This is needed for some `friend`
 * functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FORWARD_DECLS_HPP
#define MLPACK_METHODS_ANN_FORWARD_DECLS_HPP

namespace mlpack {

// See ffn.hpp.
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
class FFN;

// See rnn.hpp.
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
class RNN;

} // namespace mlpack

#endif
