/**
 * @file forward_decls.hpp
 * @author Ryan Curtin
 *
 * Forward declarations of network types.  This is needed for some `friend`
 * functionality.
 */
#ifndef MLPACK_METHODS_ANN_FORWARD_DECLS_HPP
#define MLPACK_METHODS_ANN_FORWARD_DECLS_HPP

namespace mlpack {
namespace ann {

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

} // namespace ann
} // namespace mlpack

#endif
