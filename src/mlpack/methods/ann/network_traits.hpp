/**
 * @file network_traits.hpp
 * @author Marcus Edel
 *
 * NetworkTraits class, a template class to get information about various
 * networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NETWORK_TRAITS_HPP
#define MLPACK_METHODS_ANN_NETWORK_TRAITS_HPP

namespace mlpack {
namespace ann {

/**
 * This is a template class that can provide information about various
 * networks. By default, this class will provide the weakest possible
 * assumptions on networks, and each network should override values as
 * necessary. If a network doesn't need to override a value, then there's no
 * need to write a NetworkTraits specialization for that class.
 */
template<typename NetworkType>
class NetworkTraits
{
 public:
  /**
   * This is true if the network is a feed forward neural network.
   */
  static const bool IsFNN = false;

  /**
   * This is true if the network is a recurrent neural network.
   */
  static const bool IsRNN = false;

  /**
   * This is true if the network is a convolutional neural network.
   */
  static const bool IsCNN = false;

  /**
   * This is true if the network is a sparse autoencoder.
   */
  static const bool IsSAE = false;
};

} // namespace ann
} // namespace mlpack

#endif

