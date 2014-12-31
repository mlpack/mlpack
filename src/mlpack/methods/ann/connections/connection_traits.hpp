/**
 * @file connection_traits.hpp
 * @author Marcus Edel
 *
 * ConnectionTraits class, a template class to get information about various
 * layers.
 */
#ifndef __MLPACK_METHOS_ANN_CONNECTIONS_CONNECTION_TRAITS_HPP
#define __MLPACK_METHOS_ANN_CONNECTIONS_CONNECTION_TRAITS_HPP

namespace mlpack {
namespace ann {

/**
 * This is a template class that can provide information about various
 * connections. By default, this class will provide the weakest possible
 * assumptions on connection, and each connection should override values as
 * necessary. If a connection doesn't need to override a value, then there's no
 * need to write a ConnectionTraits specialization for that class.
 */
template<typename ConnectionType>
class ConnectionTraits
{
 public:
  /**
   * This is true if the connection is a self connection.
   */
  static const bool IsSelfConnection = false;

  /**
   * This is true if the connection is a fullself connection.
   */
  static const bool IsFullselfConnection = false;
};

}; // namespace ann
}; // namespace mlpack

#endif
