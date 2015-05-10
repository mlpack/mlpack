/**
 * @file network_traits.hpp
 * @author Marcus Edel
 *
 * NetworkTraits class, a template class to get information about various
 * networks.
 */
#ifndef __MLPACK_METHODS_ANN_NETWORK_TRAITS_HPP
#define __MLPACK_METHODS_ANN_NETWORK_TRAITS_HPP

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
};

}; // namespace ann
}; // namespace mlpack

#endif

